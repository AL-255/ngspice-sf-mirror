/**********
Native Spectre (.scs) compatibility translator for ngspice.
Portable C99 implementation supporting commercial Spectre PDK model decks:
  - simulator lang=spectre [insensitive=yes] / simulator lang=spice switching
  - // and * comments, trailing \ and leading + line continuations
  - library / endlibrary, section / endsection -> .lib / .endl
  - include "file" [section=sec] -> .include / .lib
  - parameters (global, section, and subckt-local, incl. bare formal params)
  - statistics { ... } blocks (nominal corner pass-through)
  - inline subckt / subckt / ends (Spectre m-multiplicity semantics)
  - binned (`model name type { 0: ... 1: ... }`) and unbinned models
    (bsim4, bsim3v3, mos1..3, resistor, capacitor, diode, bjt)
  - Spectre device instances `inst (n1 n2 ...) master k=v ...`
  - ahdl_include "file.va" with automatic OSDI resolution/compilation/loading
**********/

#include "ngspice/ngspice.h"
#include "ngspice/compatmode.h"
#include "ngspice/cpdefs.h"
#include "ngspice/ftedefs.h"
#include "ngspice/fteext.h"
#include "ngspice/devdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/stringutil.h"
#include "inp_spectre.h"

#ifdef OSDI
#include "../spicelib/devices/dev.h" /* load_osdi() */
#endif

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#if !defined(_WIN32)
#include <unistd.h>
#endif

/* =========================================================================
 * Dynamic string builder & line array helpers
 * ========================================================================= */

typedef struct {
    char  *buf;
    size_t len;
    size_t cap;
} SpeBuf;

static void sb_init(SpeBuf *b)
{
    b->cap = 256;
    b->len = 0;
    b->buf = TMALLOC(char, b->cap);
    b->buf[0] = '\0';
}

static void sb_reserve(SpeBuf *b, size_t extra)
{
    if (b->len + extra + 1 <= b->cap)
        return;
    size_t ncap = b->cap ? b->cap * 2 : 256;
    while (ncap < b->len + extra + 1)
        ncap *= 2;
    b->buf = TREALLOC(char, b->buf, ncap);
    b->cap = ncap;
}

static void sb_putc(SpeBuf *b, char c)
{
    sb_reserve(b, 1);
    b->buf[b->len++] = c;
    b->buf[b->len] = '\0';
}

static void sb_puts(SpeBuf *b, const char *s)
{
    size_t n = strlen(s);
    sb_reserve(b, n);
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

static void sb_putn(SpeBuf *b, const char *s, size_t n)
{
    sb_reserve(b, n);
    memcpy(b->buf + b->len, s, n);
    b->len += n;
    b->buf[b->len] = '\0';
}

static void sb_free(SpeBuf *b)
{
    tfree(b->buf);
    b->len = b->cap = 0;
}

typedef struct {
    char **lines;
    size_t count;
    size_t cap;
} SpeLines;

static void sl_init(SpeLines *sl)
{
    sl->cap = 256;
    sl->count = 0;
    sl->lines = TMALLOC(char *, sl->cap);
}

static void sl_push_owned(SpeLines *sl, char *line_with_nl)
{
    if (sl->count >= sl->cap) {
        sl->cap *= 2;
        sl->lines = TREALLOC(char *, sl->lines, sl->cap);
    }
    sl->lines[sl->count++] = line_with_nl;
}

static void sl_emit(SpeLines *sl, const char *text)
{
    size_t n = strlen(text);
    bool has_nl = (n > 0 && text[n - 1] == '\n');
    char *copy_s = TMALLOC(char, n + (has_nl ? 1 : 2));
    memcpy(copy_s, text, n);
    if (!has_nl)
        copy_s[n++] = '\n';
    copy_s[n] = '\0';
    sl_push_owned(sl, copy_s);
}

/* =========================================================================
 * Symbol table: tracks models, subckts, and OSDI Verilog-A modules
 * ========================================================================= */

typedef enum {
    SPE_KIND_UNKNOWN = 0,
    SPE_KIND_MOS,
    SPE_KIND_RES,
    SPE_KIND_CAP,
    SPE_KIND_DIO,
    SPE_KIND_BJT,
    SPE_KIND_IND,
    SPE_KIND_SUBCKT,
    SPE_KIND_OSDI
} SpeKind;

typedef struct {
    char   *name;
    SpeKind kind;
    int     scope; /* 0 = global/file, 1 = local to current subckt */
} SpeSym;

typedef struct {
    SpeSym *items;
    size_t  count;
    size_t  cap;
} SpeSymTab;

static void sym_init(SpeSymTab *t)
{
    t->cap = 128;
    t->count = 0;
    t->items = TMALLOC(SpeSym, t->cap);
}

static void sym_free(SpeSymTab *t)
{
    for (size_t i = 0; i < t->count; i++)
        tfree(t->items[i].name);
    tfree(t->items);
    t->count = t->cap = 0;
}

static void sym_add(SpeSymTab *t, const char *name, SpeKind kind, int scope)
{
    if (!name || !*name)
        return;
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].scope == scope && strcasecmp(t->items[i].name, name) == 0) {
            t->items[i].kind = kind;
            return;
        }
    }
    if (t->count >= t->cap) {
        t->cap *= 2;
        t->items = TREALLOC(SpeSym, t->items, t->cap);
    }
    t->items[t->count].name = copy(name);
    t->items[t->count].kind = kind;
    t->items[t->count].scope = scope;
    t->count++;
}

static void sym_clear_local(SpeSymTab *t)
{
    size_t out = 0;
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].scope == 0) {
            if (out != i)
                t->items[out] = t->items[i];
            out++;
        } else {
            tfree(t->items[i].name);
        }
    }
    t->count = out;
}

static SpeKind sym_lookup(const SpeSymTab *t, const char *name)
{
    /* Prefer local scope (e.g., local BJT model sharing name with enclosing subckt) */
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].scope == 1 && strcasecmp(t->items[i].name, name) == 0)
            return t->items[i].kind;
    }
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].scope == 0 && strcasecmp(t->items[i].name, name) == 0)
            return t->items[i].kind;
    }
    return SPE_KIND_UNKNOWN;
}

/* =========================================================================
 * OSDI / Verilog-A helper (ahdl_include)
 * ========================================================================= */

#define SPE_MAX_LOADED_OSDI 128
static char *spe_loaded_osdi[SPE_MAX_LOADED_OSDI];
static int   spe_num_loaded_osdi = 0;

static char *spe_warned_va[SPE_MAX_LOADED_OSDI];
static int   spe_num_warned_va = 0;

static bool spe_osdi_already_loaded(const char *path)
{
    for (int i = 0; i < spe_num_loaded_osdi; i++) {
        if (strcmp(spe_loaded_osdi[i], path) == 0)
            return TRUE;
    }
    return FALSE;
}

static void spe_mark_osdi_loaded(const char *path)
{
    if (spe_num_loaded_osdi < SPE_MAX_LOADED_OSDI)
        spe_loaded_osdi[spe_num_loaded_osdi++] = copy(path);
}

static bool spe_va_already_warned(const char *va_rel)
{
    for (int i = 0; i < spe_num_warned_va; i++) {
        if (strcmp(spe_warned_va[i], va_rel) == 0)
            return TRUE;
    }
    if (spe_num_warned_va < SPE_MAX_LOADED_OSDI)
        spe_warned_va[spe_num_warned_va++] = copy(va_rel);
    return FALSE;
}

/* Normalize common Cadence Spectre Verilog-A extensions into strict IEEE 1364 /
 * OpenVAF-compatible Verilog-A (only used as fallback when raw openvaf fails):
 *  1. Strip mid-declaration attributes `parameter real (* ... *) name`
 *  2. Replace `limexp(` with a C1-smooth `analog function real __va_limexp`
 *     so conditional `if (...)` branches compile under OpenVAF's LRM rules. */
static bool spe_normalize_va_for_openvaf(const char *in_path, const char *out_path)
{
    FILE *fi = fopen(in_path, "rb");
    if (!fi)
        return FALSE;
    fseek(fi, 0, SEEK_END);
    long fsz = ftell(fi);
    if (fsz <= 0 || fsz > 16 * 1024 * 1024) {
        fclose(fi);
        return FALSE;
    }
    fseek(fi, 0, SEEK_SET);
    char *src = TMALLOC(char, (size_t)fsz + 1);
    size_t nread = fread(src, 1, (size_t)fsz, fi);
    fclose(fi);
    src[nread] = '\0';

    bool has_limexp = (strstr(src, "limexp") != NULL);
    bool has_sqrt = (strstr(src, "sqrt") != NULL);
    bool injected_helper = FALSE;

    SpeBuf out;
    sb_init(&out);

    static const char limexp_helper[] =
        "\nanalog function real __va_limexp;\n"
        "    input x;\n"
        "    real x;\n"
        "    begin\n"
        "        if (x > 80.0)\n"
        "            __va_limexp = exp(80.0) * (1.0 + (x - 80.0));\n"
        "        else if (x < -80.0)\n"
        "            __va_limexp = exp(-80.0);\n"
        "        else\n"
        "            __va_limexp = exp(x);\n"
        "    end\n"
        "endfunction\n";

    static const char sqrt_helper[] =
        "\nanalog function real __va_sqrt;\n"
        "    input x;\n"
        "    real x;\n"
        "    begin\n"
        "        if (x > 1.0e-300)\n"
        "            __va_sqrt = sqrt(x);\n"
        "        else\n"
        "            __va_sqrt = 0.0;\n"
        "    end\n"
        "endfunction\n";

    const char *p = src;
    while (*p) {
        /* 1. Match `parameter [real|integer] (* ... *)` -> strip mid-declaration (* ... *) */
        if (strncmp(p, "parameter", 9) == 0 &&
            (p == src || !(isalnum((unsigned char)p[-1]) || p[-1] == '_')) &&
            isspace((unsigned char)p[9])) {
            const char *q = p + 9;
            while (*q && isspace((unsigned char)*q)) q++;
            const char *kw_end = NULL;
            if (strncmp(q, "real", 4) == 0 && isspace((unsigned char)q[4]))
                kw_end = q + 4;
            else if (strncmp(q, "integer", 7) == 0 && isspace((unsigned char)q[7]))
                kw_end = q + 7;
            if (kw_end) {
                const char *r = kw_end;
                while (*r && isspace((unsigned char)*r)) r++;
                if (r[0] == '(' && r[1] == '*') {
                    const char *end_attr = strstr(r + 2, "*)");
                    if (end_attr) {
                        sb_putn(&out, p, (size_t)(kw_end - p));
                        sb_putc(&out, ' ');
                        p = end_attr + 2;
                        continue;
                    }
                }
            }
        }

        /* 2. Inject __va_limexp / __va_sqrt before `analog begin` */
        if ((has_limexp || has_sqrt) && !injected_helper &&
            strncmp(p, "analog", 6) == 0 &&
            (p == src || !(isalnum((unsigned char)p[-1]) || p[-1] == '_')) &&
            isspace((unsigned char)p[6])) {
            const char *q = p + 6;
            while (*q && isspace((unsigned char)*q)) q++;
            if (strncmp(q, "begin", 5) == 0 &&
                !(isalnum((unsigned char)q[5]) || q[5] == '_')) {
                if (has_limexp)
                    sb_puts(&out, limexp_helper);
                if (has_sqrt)
                    sb_puts(&out, sqrt_helper);
                injected_helper = TRUE;
            }
        }

        /* 3. Replace `limexp(` with `__va_limexp(` */
        if (has_limexp && strncmp(p, "limexp", 6) == 0 &&
            (p == src || !(isalnum((unsigned char)p[-1]) || p[-1] == '_'))) {
            const char *q = p + 6;
            while (*q && isspace((unsigned char)*q)) q++;
            if (*q == '(') {
                sb_puts(&out, "__va_limexp");
                p += 6;
                continue;
            }
        }

        /* 4. Replace `sqrt(` with `__va_sqrt(` to avoid 0*Inf AD NaN at x=0 */
        if (has_sqrt && strncmp(p, "sqrt", 4) == 0 &&
            (p == src || !(isalnum((unsigned char)p[-1]) || p[-1] == '_'))) {
            const char *q = p + 4;
            while (*q && isspace((unsigned char)*q)) q++;
            if (*q == '(') {
                sb_puts(&out, "__va_sqrt");
                p += 4;
                continue;
            }
        }

        sb_putc(&out, *p++);
    }

    tfree(src);
    FILE *fo = fopen(out_path, "wb");
    if (!fo) {
        sb_free(&out);
        return FALSE;
    }
    fwrite(out.buf, 1, out.len, fo);
    fclose(fo);
    sb_free(&out);
    return TRUE;
}

static uint64_t spe_fnv1a_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f)
        return 0;
    uint64_t h = 14695981039346656037ULL;
    unsigned char buf[4096];
    size_t n;
    while ((n = fread(buf, 1, sizeof(buf), f)) > 0) {
        for (size_t i = 0; i < n; i++) {
            h ^= (uint64_t)buf[i];
            h *= 1099511628211ULL;
        }
    }
    fclose(f);
    return h;
}

static bool spe_file_exists(const char *path)
{
    struct stat st;
    return (path && stat(path, &st) == 0 && S_ISREG(st.st_mode));
}

/* Scan a Verilog-A file for `module <name> (` declarations and register them as OSDI */
static void spe_scan_va_modules(const char *va_path, SpeSymTab *syms)
{
    FILE *f = fopen(va_path, "r");
    if (!f)
        return;
    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (*p && isspace((unsigned char)*p)) p++;
        if (strncmp(p, "module", 6) == 0 && isspace((unsigned char)p[6])) {
            p += 6;
            while (*p && isspace((unsigned char)*p)) p++;
            char mod[256];
            size_t k = 0;
            while (*p && (isalnum((unsigned char)*p) || *p == '_') && k + 1 < sizeof(mod))
                mod[k++] = *p++;
            mod[k] = '\0';
            if (k > 0)
                sym_add(syms, mod, SPE_KIND_OSDI, 0);
        }
    }
    fclose(f);
}

/* Handle `ahdl_include "foo.va"` by locating/compiling/loading foo.osdi */
static void spe_handle_ahdl_include(const char *va_rel, const char *dir_name, SpeSymTab *syms)
{
    SpeBuf va_path;
    sb_init(&va_path);
    if (va_rel[0] == '/' || !dir_name || !*dir_name) {
        sb_puts(&va_path, va_rel);
    } else {
        sb_puts(&va_path, dir_name);
        sb_putc(&va_path, '/');
        sb_puts(&va_path, va_rel);
    }

    /* Always scan module names so instance translation recognizes them */
    spe_scan_va_modules(va_path.buf, syms);

#ifdef OSDI
    /* Derive basename stem without .va */
    const char *base = strrchr(va_path.buf, '/');
    base = base ? base + 1 : va_path.buf;
    const char *dot = strrchr(base, '.');
    size_t stem_len = dot ? (size_t)(dot - base) : strlen(base);
    SpeBuf stem;
    sb_init(&stem);
    sb_putn(&stem, base, stem_len);

    SpeBuf osdi_path;
    sb_init(&osdi_path);

    /* 1. Sibling <dir>/<stem>.osdi */
    if (dir_name && *dir_name && va_rel[0] != '/') {
        sb_puts(&osdi_path, dir_name);
        sb_putc(&osdi_path, '/');
        sb_puts(&osdi_path, stem.buf);
        sb_puts(&osdi_path, ".osdi");
        if (!spe_file_exists(osdi_path.buf)) {
            osdi_path.len = 0; osdi_path.buf[0] = '\0';
        }
    } else {
        const char *sl = strrchr(va_path.buf, '/');
        if (sl) {
            sb_putn(&osdi_path, va_path.buf, (size_t)(sl - va_path.buf));
            sb_putc(&osdi_path, '/');
            sb_puts(&osdi_path, stem.buf);
            sb_puts(&osdi_path, ".osdi");
            if (!spe_file_exists(osdi_path.buf)) {
                osdi_path.len = 0; osdi_path.buf[0] = '\0';
            }
        }
    }

    /* 2. $NGSPICE_OSDI_DIR/<stem>.osdi */
    if (osdi_path.len == 0) {
        const char *odir = getenv("NGSPICE_OSDI_DIR");
        if (odir && *odir) {
            sb_puts(&osdi_path, odir);
            sb_putc(&osdi_path, '/');
            sb_puts(&osdi_path, stem.buf);
            sb_puts(&osdi_path, ".osdi");
            if (!spe_file_exists(osdi_path.buf)) {
                osdi_path.len = 0; osdi_path.buf[0] = '\0';
            }
        }
    }

    /* 3. Cached $HOME/.cache/ngspice/osdi/<stem>_<hash>.osdi (or auto-compile via openvaf) */
    if (osdi_path.len == 0 && spe_file_exists(va_path.buf)) {
        const char *home = getenv("HOME");
        if (home && *home) {
            uint64_t h = spe_fnv1a_file(va_path.buf);
            char hhex[32];
            snprintf(hhex, sizeof(hhex), "%016llx", (unsigned long long)h);

            SpeBuf cache_dir, cand;
            sb_init(&cache_dir);
            sb_puts(&cache_dir, home);
            sb_puts(&cache_dir, "/.cache/ngspice/osdi");

            sb_init(&cand);
            sb_puts(&cand, cache_dir.buf);
            sb_putc(&cand, '/');
            sb_puts(&cand, stem.buf);
            sb_putc(&cand, '_');
            sb_puts(&cand, hhex);
            sb_puts(&cand, ".osdi");

            if (spe_file_exists(cand.buf)) {
                sb_puts(&osdi_path, cand.buf);
            } else {
                /* Try compiling on demand with $OPENVAF, openvaf-r, or openvaf */
                const char *ov = getenv("OPENVAF");
                SpeBuf cmd;
                sb_init(&cmd);
                sb_puts(&cmd, "mkdir -p \"");
                sb_puts(&cmd, cache_dir.buf);
                sb_puts(&cmd, "\" && ");
                if (ov && *ov) {
                    sb_putc(&cmd, '"'); sb_puts(&cmd, ov);
                    sb_puts(&cmd, "\" \""); sb_puts(&cmd, va_path.buf);
                    sb_puts(&cmd, "\" -o \""); sb_puts(&cmd, cand.buf);
                    sb_puts(&cmd, "\" >/dev/null 2>&1");
                } else {
                    sb_puts(&cmd, "(command -v openvaf-r >/dev/null 2>&1 && openvaf-r \"");
                    sb_puts(&cmd, va_path.buf); sb_puts(&cmd, "\" -o \""); sb_puts(&cmd, cand.buf);
                    sb_puts(&cmd, "\" >/dev/null 2>&1 || command -v openvaf >/dev/null 2>&1 && openvaf \"");
                    sb_puts(&cmd, va_path.buf); sb_puts(&cmd, "\" -o \""); sb_puts(&cmd, cand.buf);
                    sb_puts(&cmd, "\" >/dev/null 2>&1)");
                }
                if (system(cmd.buf) == 0 && spe_file_exists(cand.buf)) {
                    sb_puts(&osdi_path, cand.buf);
                } else {
                    /* Retry after normalizing Cadence VA extensions (mid-decl attributes / limexp-in-cond) */
                    SpeBuf norm_va;
                    sb_init(&norm_va);
                    sb_puts(&norm_va, cache_dir.buf);
                    sb_putc(&norm_va, '/');
                    sb_puts(&norm_va, stem.buf);
                    sb_putc(&norm_va, '_');
                    sb_puts(&norm_va, hhex);
                    sb_puts(&norm_va, "_compat.va");

                    if (spe_normalize_va_for_openvaf(va_path.buf, norm_va.buf)) {
                        cmd.len = 0; cmd.buf[0] = '\0';
                        if (ov && *ov) {
                            sb_putc(&cmd, '"'); sb_puts(&cmd, ov);
                            sb_puts(&cmd, "\" \""); sb_puts(&cmd, norm_va.buf);
                            sb_puts(&cmd, "\" -o \""); sb_puts(&cmd, cand.buf);
                            sb_puts(&cmd, "\" >/dev/null 2>&1");
                        } else {
                            sb_puts(&cmd, "(command -v openvaf-r >/dev/null 2>&1 && openvaf-r \"");
                            sb_puts(&cmd, norm_va.buf); sb_puts(&cmd, "\" -o \""); sb_puts(&cmd, cand.buf);
                            sb_puts(&cmd, "\" >/dev/null 2>&1 || command -v openvaf >/dev/null 2>&1 && openvaf \"");
                            sb_puts(&cmd, norm_va.buf); sb_puts(&cmd, "\" -o \""); sb_puts(&cmd, cand.buf);
                            sb_puts(&cmd, "\" >/dev/null 2>&1)");
                        }
                        if (system(cmd.buf) == 0 && spe_file_exists(cand.buf))
                            sb_puts(&osdi_path, cand.buf);
                        remove(norm_va.buf);
                    }
                    sb_free(&norm_va);
                }
                sb_free(&cmd);
            }
            sb_free(&cand);
            sb_free(&cache_dir);
        }
    }

    if (osdi_path.len > 0) {
        if (!spe_osdi_already_loaded(osdi_path.buf)) {
            if (load_osdi(osdi_path.buf) == 0) {
                spe_mark_osdi_loaded(osdi_path.buf);
            } else {
                fprintf(stderr, "Warning: Spectre ahdl_include failed to load OSDI '%s'\n", osdi_path.buf);
            }
        }
    } else {
        if (!spe_va_already_warned(va_rel)) {
            fprintf(stderr,
                    "Warning: Spectre ahdl_include '%s': no compiled '%s.osdi' found "
                    "(set NGSPICE_OSDI_DIR or install openvaf/openvaf-r on PATH)\n",
                    va_rel, stem.buf);
        }
    }
    sb_free(&osdi_path);
    sb_free(&stem);
#endif
    sb_free(&va_path);
}

/* Query ft_sim->devices[] to check whether `pname` is an instance param or model param
 * for a loaded OSDI device `modname`. Returns: 1 = instance param, 2 = model param, 0 = unknown. */
static int spe_osdi_param_scope(const char *modname, const char *pname)
{
#ifdef OSDI
    if (!ft_sim || !modname || !pname)
        return 0;
    for (int t = 0; t < ft_sim->numDevices; t++) {
        IFdevice *dev = ft_sim->devices[t];
        if (!dev || !dev->registry_entry || !dev->name)
            continue;
        if (strcasecmp(dev->name, modname) != 0)
            continue;
        if (dev->instanceParms && dev->numInstanceParms) {
            int ni = *(dev->numInstanceParms);
            for (int i = 0; i < ni; i++) {
                if (dev->instanceParms[i].keyword &&
                    strcasecmp(dev->instanceParms[i].keyword, pname) == 0)
                    return 1;
            }
        }
        if (dev->modelParms && dev->numModelParms) {
            int nm = *(dev->numModelParms);
            for (int i = 0; i < nm; i++) {
                if (dev->modelParms[i].keyword &&
                    strcasecmp(dev->modelParms[i].keyword, pname) == 0)
                    return 2;
            }
        }
        return 0;
    }
#else
    (void)modname;
    (void)pname;
#endif
    return 0;
}

/* Also check if `name` is already registered as an OSDI device in ft_sim */
static bool spe_is_loaded_osdi_device(const char *name)
{
#ifdef OSDI
    if (!ft_sim || !name)
        return FALSE;
    for (int t = 0; t < ft_sim->numDevices; t++) {
        IFdevice *dev = ft_sim->devices[t];
        if (dev && dev->registry_entry && dev->name && strcasecmp(dev->name, name) == 0)
            return TRUE;
    }
#else
    (void)name;
#endif
    return FALSE;
}

/* =========================================================================
 * Lexical & expression helpers
 * ========================================================================= */

static bool is_ident_start(char c)
{
    return isalpha((unsigned char)c) || c == '_';
}

static bool is_ident_char(char c)
{
    return isalnum((unsigned char)c) || c == '_' || c == '$';
}

static void strip_inline_comment(char *s)
{
    bool in_sq = FALSE, in_dq = FALSE;
    for (char *p = s; *p; p++) {
        if (*p == '\'' && !in_dq) in_sq = !in_sq;
        else if (*p == '"' && !in_sq) in_dq = !in_dq;
        else if (!in_sq && !in_dq && p[0] == '/' && p[1] == '/') {
            *p = '\0';
            break;
        }
    }
    /* Trim trailing whitespace / CR / LF */
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == '\n' || s[n - 1] == '\r' ||
                     s[n - 1] == ' ' || s[n - 1] == '\t'))
        s[--n] = '\0';
}

static bool is_spe_operator_char(char c)
{
    return (c == '+' || c == '-' || c == '*' || c == '/' ||
            c == '?' || c == ':' || c == '|' || c == '&' ||
            c == '=' || c == '!' || c == '<' || c == '>' || c == '^');
}

/* Check if a string starts with an identifier followed by optional spaces and '=' (not '==') */
static bool looks_like_kv_start(const char *p)
{
    while (*p && isspace((unsigned char)*p)) p++;
    if (!is_ident_start(*p))
        return FALSE;
    while (*p && is_ident_char(*p)) p++;
    while (*p && isspace((unsigned char)*p)) p++;
    return (*p == '=' && p[1] != '=');
}

/* Check if a value string is a plain literal number (incl. SPICE suffix, optional leading +/-)
 * or a single bare identifier (no operators/function calls). */
static bool is_simple_val(const char *v)
{
    if (!v || !*v)
        return TRUE;
    const char *p = v;
    /* Check plain identifier */
    if (is_ident_start(*p)) {
        while (*p && is_ident_char(*p)) p++;
        return (*p == '\0');
    }
    /* Check signed number with optional exponent and engineering suffix */
    if (*p == '+' || *p == '-')
        p++;
    bool has_digit = FALSE;
    while (*p && isdigit((unsigned char)*p)) { has_digit = TRUE; p++; }
    if (*p == '.') {
        p++;
        while (*p && isdigit((unsigned char)*p)) { has_digit = TRUE; p++; }
    }
    if (!has_digit)
        return FALSE;
    if (*p == 'e' || *p == 'E') {
        p++;
        if (*p == '+' || *p == '-') p++;
        bool ed = FALSE;
        while (*p && isdigit((unsigned char)*p)) { ed = TRUE; p++; }
        if (!ed) return FALSE;
    }
    /* Engineering suffix (u, n, p, f, a, k, m, g, t, meg, mil) */
    while (*p && isalpha((unsigned char)*p)) p++;
    return (*p == '\0');
}

static bool is_plain_number(const char *v)
{
    if (!v || !*v)
        return FALSE;
    const char *p = v;
    if (*p == '+' || *p == '-') p++;
    if (!isdigit((unsigned char)*p) && *p != '.')
        return FALSE;
    return is_simple_val(v);
}

/* Format a Spectre expression into ngspice numparam syntax:
 * - removes interior whitespace
 * - replaces standalone `temp` with `temper`
 * - wraps in `{...}` when `force_braces` or expression is non-simple */
static void spe_format_expr(SpeBuf *out, const char *expr, bool force_braces)
{
    SpeBuf clean;
    sb_init(&clean);

    const char *p = expr;
    while (*p) {
        if (isspace((unsigned char)*p)) {
            p++;
            continue;
        }
        if (is_ident_start(*p)) {
            const char *start = p;
            while (*p && is_ident_char(*p)) p++;
            size_t len = (size_t)(p - start);
            if (len == 4 && strncasecmp(start, "temp", 4) == 0 && *p != '(') {
                sb_puts(&clean, "temper");
            } else {
                sb_putn(&clean, start, len);
            }
            continue;
        }
        sb_putc(&clean, *p++);
    }

    bool need_braces = force_braces || !is_simple_val(clean.buf);
    if (need_braces && clean.len > 0 && clean.buf[0] != '{') {
        sb_putc(out, '{');
        sb_puts(out, clean.buf);
        sb_putc(out, '}');
    } else {
        sb_puts(out, clean.buf);
    }
    sb_free(&clean);
}

/* Parse next key[=val] from Spectre parameter/argument list.
 * Advances `*pp`. Returns TRUE if a key was parsed.
 * If `allow_bare` is TRUE, a standalone identifier without '=' sets `*has_val=FALSE`. */
static bool spe_next_kv(const char **pp, char *key_out, size_t key_cap,
                        SpeBuf *val_out, bool *has_val, bool allow_bare)
{
    const char *p = *pp;
    while (*p && isspace((unsigned char)*p)) p++;
    if (!*p) {
        *pp = p;
        return FALSE;
    }
    if (!is_ident_start(*p)) {
        /* Skip unexpected punctuation token */
        while (*p && !isspace((unsigned char)*p)) p++;
        *pp = p;
        return FALSE;
    }

    size_t klen = 0;
    while (*p && is_ident_char(*p)) {
        if (klen + 1 < key_cap)
            key_out[klen++] = *p;
        p++;
    }
    key_out[klen] = '\0';

    const char *after_key = p;
    while (*p && isspace((unsigned char)*p)) p++;

    if (*p != '=' || p[1] == '=') {
        if (allow_bare) {
            *has_val = FALSE;
            val_out->len = 0;
            val_out->buf[0] = '\0';
            *pp = after_key;
            return TRUE;
        }
        *pp = after_key;
        return FALSE;
    }

    p++; /* skip '=' */
    while (*p && isspace((unsigned char)*p)) p++;
    *has_val = TRUE;
    val_out->len = 0;
    val_out->buf[0] = '\0';

    int paren = 0, brack = 0;
    bool in_sq = FALSE, in_dq = FALSE;
    char last_nonws = '=';

    while (*p) {
        char c = *p;
        if (in_sq) {
            sb_putc(val_out, c);
            if (c == '\'') in_sq = FALSE;
            p++;
            continue;
        }
        if (in_dq) {
            sb_putc(val_out, c);
            if (c == '"') in_dq = FALSE;
            p++;
            continue;
        }
        if (c == '\'') { in_sq = TRUE; sb_putc(val_out, c); last_nonws = c; p++; continue; }
        if (c == '"')  { in_dq = TRUE; sb_putc(val_out, c); last_nonws = c; p++; continue; }
        if (c == '(')  { paren++; sb_putc(val_out, c); last_nonws = c; p++; continue; }
        if (c == ')')  { if (paren > 0) paren--; sb_putc(val_out, c); last_nonws = c; p++; continue; }
        if (c == '[')  { brack++; sb_putc(val_out, c); last_nonws = c; p++; continue; }
        if (c == ']')  { if (brack > 0) brack--; sb_putc(val_out, c); last_nonws = c; p++; continue; }

        if (paren == 0 && brack == 0 && isspace((unsigned char)c)) {
            const char *q = p;
            while (*q && isspace((unsigned char)*q)) q++;
            if (!*q) {
                p = q;
                break;
            }
            /* Continue across whitespace if previous token ended with an operator
             * OR next token starts with an operator (e.g., `a * b`, `(c==0) ? 1 : 2`) */
            bool prev_is_op = is_spe_operator_char(last_nonws);
            bool next_is_op = is_spe_operator_char(*q);
            if (prev_is_op || next_is_op) {
                sb_putc(val_out, ' ');
                p = q;
                continue;
            }
            /* Otherwise if next starts a new key=val (or bare formal param), stop */
            p = q;
            break;
        }

        sb_putc(val_out, c);
        last_nonws = c;
        p++;
    }

    /* Trim trailing space */
    while (val_out->len > 0 && isspace((unsigned char)val_out->buf[val_out->len - 1]))
        val_out->buf[--val_out->len] = '\0';

    *pp = p;
    return TRUE;
}

static SpeKind spectre_model_type_to_kind(const char *mtype)
{
    if (strcasecmp(mtype, "bsim4") == 0 || strcasecmp(mtype, "bsim3v3") == 0 ||
        strcasecmp(mtype, "bsim3") == 0 || strcasecmp(mtype, "mos1") == 0 ||
        strcasecmp(mtype, "mos2") == 0 || strcasecmp(mtype, "mos3") == 0 ||
        strcasecmp(mtype, "level1") == 0 || strcasecmp(mtype, "ekv") == 0 ||
        strcasecmp(mtype, "bsim4v5") == 0)
        return SPE_KIND_MOS;
    if (strcasecmp(mtype, "resistor") == 0)
        return SPE_KIND_RES;
    if (strcasecmp(mtype, "capacitor") == 0)
        return SPE_KIND_CAP;
    if (strcasecmp(mtype, "diode") == 0)
        return SPE_KIND_DIO;
    if (strcasecmp(mtype, "bjt") == 0)
        return SPE_KIND_BJT;
    if (strcasecmp(mtype, "inductor") == 0)
        return SPE_KIND_IND;
    return SPE_KIND_UNKNOWN;
}

/* =========================================================================
 * Raw file reader & logical Spectre statement joiner
 * ========================================================================= */

typedef struct {
    char **stmts;
    bool  *is_spice; /* TRUE if statement came from a `simulator lang=spice` block */
    size_t count;
    size_t cap;
} SpeStmtList;

static void st_init(SpeStmtList *st)
{
    st->cap = 256;
    st->count = 0;
    st->stmts = TMALLOC(char *, st->cap);
    st->is_spice = TMALLOC(bool, st->cap);
}

static void st_push(SpeStmtList *st, char *s, bool spice_mode)
{
    if (st->count >= st->cap) {
        st->cap *= 2;
        st->stmts = TREALLOC(char *, st->stmts, st->cap);
        st->is_spice = TREALLOC(bool, st->is_spice, st->cap);
    }
    st->stmts[st->count] = s;
    st->is_spice[st->count] = spice_mode;
    st->count++;
}

static void st_free(SpeStmtList *st)
{
    for (size_t i = 0; i < st->count; i++)
        tfree(st->stmts[i]);
    tfree(st->stmts);
    tfree(st->is_spice);
    st->count = st->cap = 0;
}

/* Read entire FILE into raw lines, strip // comments, join Spectre continuations (\ and +)
 * and brace-delimited multi-line blocks (`model ... { ... }` and `statistics { ... }`). */
static void spe_read_statements(FILE *fp, SpeStmtList *out)
{
    char linebuf[8192];
    bool in_spectre = TRUE;
    SpeBuf cur;
    sb_init(&cur);
    int brace_depth = 0;
    bool backslash_cont = FALSE;

    while (fgets(linebuf, sizeof(linebuf), fp)) {
        /* Strip UTF-8 BOM */
        char *raw = linebuf;
        if (strncmp(raw, "\xEF\xBB\xBF", 3) == 0)
            raw += 3;

        /* Trim trailing CR/LF */
        size_t rlen = strlen(raw);
        while (rlen > 0 && (raw[rlen - 1] == '\n' || raw[rlen - 1] == '\r'))
            raw[--rlen] = '\0';

        char *p = raw;
        while (*p && isspace((unsigned char)*p)) p++;

        /* Check simulator lang=... directive */
        if (strncasecmp(p, "simulator", 9) == 0 && isspace((unsigned char)p[9])) {
            if (cur.len > 0) {
                st_push(out, copy(cur.buf), !in_spectre);
                cur.len = 0; cur.buf[0] = '\0';
                brace_depth = 0; backslash_cont = FALSE;
            }
            char low[256];
            size_t i;
            for (i = 0; p[i] && i + 1 < sizeof(low); i++)
                low[i] = (char)tolower((unsigned char)p[i]);
            low[i] = '\0';
            if (strstr(low, "lang=spice"))
                in_spectre = FALSE;
            else if (strstr(low, "lang=spectre"))
                in_spectre = TRUE;
            continue;
        }

        if (!in_spectre) {
            /* Pass SPICE-language lines verbatim */
            st_push(out, copy(raw), TRUE);
            continue;
        }

        /* Spectre mode: strip // comments & ignore pure comment/blank lines */
        if (*p == '*')
            continue;
        strip_inline_comment(raw);
        p = raw;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p)
            continue;

        bool ends_backslash = FALSE;
        size_t plen = strlen(p);
        if (plen > 0 && p[plen - 1] == '\\') {
            ends_backslash = TRUE;
            p[--plen] = '\0';
            while (plen > 0 && isspace((unsigned char)p[plen - 1]))
                p[--plen] = '\0';
        }

        bool is_plus_cont = (*p == '+');
        if (is_plus_cont) {
            p++;
            while (*p && isspace((unsigned char)*p)) p++;
        }

        if (backslash_cont || is_plus_cont || brace_depth > 0) {
            if (cur.len > 0) {
                /* Inside a brace block (like binned model { 0: ... 1: ... }),
                 * preserve newline boundaries so bins remain easy to split */
                sb_putc(&cur, brace_depth > 0 && !is_plus_cont && !backslash_cont ? '\n' : ' ');
            }
            sb_puts(&cur, p);
        } else {
            if (cur.len > 0) {
                st_push(out, copy(cur.buf), FALSE);
                cur.len = 0; cur.buf[0] = '\0';
            }
            sb_puts(&cur, p);
        }

        /* Update brace depth */
        bool in_sq = FALSE, in_dq = FALSE;
        for (const char *q = p; *q; q++) {
            if (*q == '\'' && !in_dq) in_sq = !in_sq;
            else if (*q == '"' && !in_sq) in_dq = !in_dq;
            else if (!in_sq && !in_dq) {
                if (*q == '{') brace_depth++;
                else if (*q == '}' && brace_depth > 0) brace_depth--;
            }
        }
        backslash_cont = ends_backslash;

        /* Flush completed brace block immediately */
        if (brace_depth == 0 && !backslash_cont && strchr(cur.buf, '}')) {
            st_push(out, copy(cur.buf), FALSE);
            cur.len = 0; cur.buf[0] = '\0';
        }
    }

    if (cur.len > 0)
        st_push(out, copy(cur.buf), !in_spectre);
    sb_free(&cur);
}

/* =========================================================================
 * Model emission helper (binned and unbinned)
 * ========================================================================= */

static void spe_emit_model_params(SpeBuf *out, const char *mtype, const char *params,
                                  char *detected_polarity, size_t pol_cap)
{
    const char *p = params;
    char key[128];
    SpeBuf val;
    sb_init(&val);
    bool has_val = FALSE;

    while (*p) {
        if (!spe_next_kv(&p, key, sizeof(key), &val, &has_val, FALSE))
            continue;
        if (!has_val)
            continue;

        if (strcasecmp(key, "type") == 0) {
            if (detected_polarity && pol_cap > 0)
                snprintf(detected_polarity, pol_cap, "%s", val.buf);
            continue;
        }
        /* Skip Spectre-only compatibility/vector flags not recognized by SPICE models */
        if (strcasecmp(key, "coeffs") == 0 || strcasecmp(key, "compatible") == 0)
            continue;

        /* Map Spectre BJT `struct=vertical|lateral` -> SPICE `subs=1|-1` */
        if (strcasecmp(mtype, "bjt") == 0 && strcasecmp(key, "struct") == 0) {
            if (strcasecmp(val.buf, "vertical") == 0)
                sb_puts(out, " subs=1");
            else if (strcasecmp(val.buf, "lateral") == 0)
                sb_puts(out, " subs=-1");
            continue;
        }

        /* Map Spectre tref -> ngspice tnom */
        const char *out_key = key;
        if (strcasecmp(key, "tref") == 0)
            out_key = "tnom";

        sb_putc(out, ' ');
        sb_puts(out, out_key);
        sb_putc(out, '=');
        spe_format_expr(out, val.buf, FALSE);
    }

    sb_free(&val);
}

static void spe_resolve_spice_model_type(const char *mtype, const char *pol,
                                         char *spice_type, size_t st_cap,
                                         const char **extra_level)
{
    *extra_level = "";
    if (strcasecmp(mtype, "bsim4") == 0 || strcasecmp(mtype, "bsim4v5") == 0) {
        snprintf(spice_type, st_cap, "%s", (pol && (pol[0] == 'p' || pol[0] == 'P')) ? "pmos" : "nmos");
        *extra_level = " level=54";
    } else if (strcasecmp(mtype, "bsim3v3") == 0 || strcasecmp(mtype, "bsim3") == 0) {
        snprintf(spice_type, st_cap, "%s", (pol && (pol[0] == 'p' || pol[0] == 'P')) ? "pmos" : "nmos");
        *extra_level = " level=49";
    } else if (strcasecmp(mtype, "mos1") == 0 || strcasecmp(mtype, "level1") == 0) {
        snprintf(spice_type, st_cap, "%s", (pol && (pol[0] == 'p' || pol[0] == 'P')) ? "pmos" : "nmos");
        *extra_level = " level=1";
    } else if (strcasecmp(mtype, "resistor") == 0) {
        snprintf(spice_type, st_cap, "r");
    } else if (strcasecmp(mtype, "capacitor") == 0) {
        snprintf(spice_type, st_cap, "c");
    } else if (strcasecmp(mtype, "diode") == 0) {
        snprintf(spice_type, st_cap, "d");
    } else if (strcasecmp(mtype, "bjt") == 0) {
        snprintf(spice_type, st_cap, "%s",
                 (pol && (strncasecmp(pol, "pnp", 3) == 0 || pol[0] == 'p' || pol[0] == 'P')) ? "pnp" : "npn");
    } else {
        snprintf(spice_type, st_cap, "%s", mtype);
    }
}

static void spe_translate_model(const char *stmt, SpeLines *out)
{
    /* Syntax:
     *   model <name> <type> <params...>
     *   model <name> <type> { \n 0: <params...> \n 1: <params...> \n } */
    const char *p = stmt + 5; /* skip "model" */
    while (*p && isspace((unsigned char)*p)) p++;

    char mname[256], mtype[128];
    size_t k = 0;
    while (*p && !isspace((unsigned char)*p) && k + 1 < sizeof(mname)) mname[k++] = *p++;
    mname[k] = '\0';
    while (*p && isspace((unsigned char)*p)) p++;
    k = 0;
    while (*p && !isspace((unsigned char)*p) && *p != '{' && k + 1 < sizeof(mtype)) mtype[k++] = *p++;
    mtype[k] = '\0';
    while (*p && isspace((unsigned char)*p)) p++;

    if (*p != '{') {
        /* Unbinned model */
        char pol[32] = "";
        SpeBuf pbuf;
        sb_init(&pbuf);
        spe_emit_model_params(&pbuf, mtype, p, pol, sizeof(pol));

        char stype[64];
        const char *extra = "";
        spe_resolve_spice_model_type(mtype, pol, stype, sizeof(stype), &extra);

        SpeBuf line;
        sb_init(&line);
        sb_puts(&line, ".model ");
        sb_puts(&line, mname);
        sb_putc(&line, ' ');
        sb_puts(&line, stype);
        sb_puts(&line, extra);
        sb_puts(&line, pbuf.buf);
        sl_emit(out, line.buf);
        sb_free(&line);
        sb_free(&pbuf);
        return;
    }

    /* Binned model: split body inside { ... } by lines starting with `<digits>:` */
    p++; /* skip '{' */
    const char *end_brace = strrchr(p, '}');
    size_t blen = end_brace ? (size_t)(end_brace - p) : strlen(p);
    char *body = TMALLOC(char, blen + 1);
    memcpy(body, p, blen);
    body[blen] = '\0';

    /* Walk line by line inside body; collect each bin */
    char cur_bin[32] = "";
    SpeBuf bin_params;
    sb_init(&bin_params);

    char *q = body;
    while (*q) {
        char *nl = strchr(q, '\n');
        if (nl) *nl = '\0';
        char *l = q;
        while (*l && isspace((unsigned char)*l)) l++;

        /* Detect `<digits>:` at start of line */
        char *c = l;
        while (*c && isdigit((unsigned char)*c)) c++;
        if (c > l && *c == ':') {
            /* Flush previous bin */
            if (cur_bin[0]) {
                char pol[32] = "";
                SpeBuf pbuf;
                sb_init(&pbuf);
                spe_emit_model_params(&pbuf, mtype, bin_params.buf, pol, sizeof(pol));
                char stype[64];
                const char *extra = "";
                spe_resolve_spice_model_type(mtype, pol, stype, sizeof(stype), &extra);

                SpeBuf line;
                sb_init(&line);
                sb_puts(&line, ".model ");
                sb_puts(&line, mname);
                sb_putc(&line, '.');
                sb_puts(&line, cur_bin);
                sb_putc(&line, ' ');
                sb_puts(&line, stype);
                sb_puts(&line, extra);
                sb_puts(&line, pbuf.buf);
                sl_emit(out, line.buf);
                sb_free(&line);
                sb_free(&pbuf);
            }
            size_t dlen = (size_t)(c - l);
            if (dlen >= sizeof(cur_bin)) dlen = sizeof(cur_bin) - 1;
            memcpy(cur_bin, l, dlen);
            cur_bin[dlen] = '\0';
            bin_params.len = 0; bin_params.buf[0] = '\0';
            sb_puts(&bin_params, c + 1);
        } else if (*l) {
            if (bin_params.len > 0) sb_putc(&bin_params, ' ');
            sb_puts(&bin_params, l);
        }

        if (!nl) break;
        q = nl + 1;
    }

    /* Flush final bin */
    if (cur_bin[0]) {
        char pol[32] = "";
        SpeBuf pbuf;
        sb_init(&pbuf);
        spe_emit_model_params(&pbuf, mtype, bin_params.buf, pol, sizeof(pol));
        char stype[64];
        const char *extra = "";
        spe_resolve_spice_model_type(mtype, pol, stype, sizeof(stype), &extra);

        SpeBuf line;
        sb_init(&line);
        sb_puts(&line, ".model ");
        sb_puts(&line, mname);
        sb_putc(&line, '.');
        sb_puts(&line, cur_bin);
        sb_putc(&line, ' ');
        sb_puts(&line, stype);
        sb_puts(&line, extra);
        sb_puts(&line, pbuf.buf);
        sl_emit(out, line.buf);
        sb_free(&line);
        sb_free(&pbuf);
    }

    sb_free(&bin_params);
    tfree(body);
}

/* =========================================================================
 * Device instance translation (`inst (n1 n2 ...) master k=v ...`)
 * ========================================================================= */

static unsigned long spe_va_uid = 0;

static void spe_translate_instance(const char *stmt, const SpeSymTab *syms, SpeLines *out)
{
    const char *lp = strchr(stmt, '(');
    if (!lp) {
        /* Fallback: emit as comment if not a valid instance */
        SpeBuf c; sb_init(&c);
        sb_puts(&c, "* [spectre-unparsed] ");
        sb_puts(&c, stmt);
        sl_emit(out, c.buf);
        sb_free(&c);
        return;
    }
    const char *rp = strchr(lp + 1, ')');
    if (!rp)
        return;

    /* Extract instance name */
    char inst[256];
    const char *p = stmt;
    while (p < lp && isspace((unsigned char)*p)) p++;
    size_t k = 0;
    while (p < lp && !isspace((unsigned char)*p) && k + 1 < sizeof(inst))
        inst[k++] = *p++;
    inst[k] = '\0';

    /* Extract node list inside (...) */
    SpeBuf nodes;
    sb_init(&nodes);
    p = lp + 1;
    while (p < rp) {
        while (p < rp && isspace((unsigned char)*p)) p++;
        if (p >= rp) break;
        if (nodes.len > 0) sb_putc(&nodes, ' ');
        while (p < rp && !isspace((unsigned char)*p))
            sb_putc(&nodes, *p++);
    }

    /* Extract master name */
    p = rp + 1;
    while (*p && isspace((unsigned char)*p)) p++;
    char master[256];
    k = 0;
    while (*p && !isspace((unsigned char)*p) && k + 1 < sizeof(master))
        master[k++] = *p++;
    master[k] = '\0';
    while (*p && isspace((unsigned char)*p)) p++;

    /* Built-in Spectre primitive check */
    bool is_prim_res = (strcasecmp(master, "resistor") == 0);
    bool is_prim_cap = (strcasecmp(master, "capacitor") == 0);
    bool is_prim_ind = (strcasecmp(master, "inductor") == 0);
    bool is_prim_vsrc = (strcasecmp(master, "vsource") == 0);
    bool is_prim_isrc = (strcasecmp(master, "isource") == 0);
    bool is_prim_vcvs = (strcasecmp(master, "vcvs") == 0);
    bool is_prim_bsrc = (strcasecmp(master, "bsource") == 0);

    SpeKind kind = SPE_KIND_UNKNOWN;
    if (is_prim_res) kind = SPE_KIND_RES;
    else if (is_prim_cap) kind = SPE_KIND_CAP;
    else if (is_prim_ind) kind = SPE_KIND_IND;
    else {
        kind = sym_lookup(syms, master);
        if (kind == SPE_KIND_UNKNOWN && spe_is_loaded_osdi_device(master))
            kind = SPE_KIND_OSDI;
    }

    /* Handle OSDI Verilog-A module instantiation (`n_<inst> nodes __spe_va_<uid>`) */
    if (kind == SPE_KIND_OSDI) {
        char mname[128];
        snprintf(mname, sizeof(mname), "__spe_va_%lu", ++spe_va_uid);

        SpeBuf mod_line, inst_line;
        sb_init(&mod_line);
        sb_init(&inst_line);
        sb_puts(&mod_line, ".model ");
        sb_puts(&mod_line, mname);
        sb_putc(&mod_line, ' ');
        sb_puts(&mod_line, master);

        sb_puts(&inst_line, "n_");
        sb_puts(&inst_line, inst);
        sb_putc(&inst_line, ' ');
        sb_puts(&inst_line, nodes.buf);
        sb_putc(&inst_line, ' ');
        sb_puts(&inst_line, mname);

        char key[128];
        SpeBuf val;
        sb_init(&val);
        bool has_val = FALSE;
        const char *q = p;
        while (*q) {
            if (!spe_next_kv(&q, key, sizeof(key), &val, &has_val, FALSE) || !has_val)
                continue;
            const char *okey = (strcasecmp(key, "trise") == 0) ? "dtemp" : key;
            int sc = spe_osdi_param_scope(master, okey);
            if (sc == 1 || strcasecmp(okey, "m") == 0 || strcasecmp(okey, "temp") == 0) {
                sb_putc(&inst_line, ' ');
                sb_puts(&inst_line, okey);
                sb_putc(&inst_line, '=');
                spe_format_expr(&inst_line, val.buf, FALSE);
            } else {
                sb_putc(&mod_line, ' ');
                sb_puts(&mod_line, okey);
                sb_putc(&mod_line, '=');
                spe_format_expr(&mod_line, val.buf, FALSE);
            }
        }
        sl_emit(out, mod_line.buf);
        sl_emit(out, inst_line.buf);
        sb_free(&val);
        sb_free(&mod_line);
        sb_free(&inst_line);
        sb_free(&nodes);
        return;
    }

    /* Handle primitive vcvs / bsource / vsource / isource if used in Spectre testbenches */
    if (is_prim_vcvs) {
        SpeBuf line; sb_init(&line);
        if (tolower((unsigned char)inst[0]) == 'e') sb_puts(&line, inst);
        else { sb_puts(&line, "e_"); sb_puts(&line, inst); }
        sb_putc(&line, ' ');
        sb_puts(&line, nodes.buf);
        char key[128]; SpeBuf val; sb_init(&val); bool hv = FALSE;
        const char *q = p;
        const char *gain_v = "1";
        SpeBuf gbuf; sb_init(&gbuf);
        while (*q) {
            if (spe_next_kv(&q, key, sizeof(key), &val, &hv, FALSE) && hv && strcasecmp(key, "gain") == 0) {
                sb_puts(&gbuf, val.buf);
                gain_v = gbuf.buf;
            }
        }
        sb_putc(&line, ' ');
        spe_format_expr(&line, gain_v, FALSE);
        sl_emit(out, line.buf);
        sb_free(&gbuf); sb_free(&val); sb_free(&line); sb_free(&nodes);
        return;
    }
    if (is_prim_bsrc) {
        SpeBuf line; sb_init(&line);
        if (tolower((unsigned char)inst[0]) == 'b') sb_puts(&line, inst);
        else { sb_puts(&line, "b_"); sb_puts(&line, inst); }
        sb_putc(&line, ' ');
        sb_puts(&line, nodes.buf); sb_putc(&line, ' ');
        sb_puts(&line, p);
        sl_emit(out, line.buf);
        sb_free(&line); sb_free(&nodes);
        return;
    }
    if (is_prim_vsrc || is_prim_isrc) {
        SpeBuf line; sb_init(&line);
        char spfx = is_prim_vsrc ? 'v' : 'i';
        if (tolower((unsigned char)inst[0]) == spfx) sb_puts(&line, inst);
        else { sb_putc(&line, spfx); sb_putc(&line, '_'); sb_puts(&line, inst); }
        sb_putc(&line, ' ');
        sb_puts(&line, nodes.buf);
        char key[128]; SpeBuf val; sb_init(&val); bool hv = FALSE;
        const char *q = p;
        while (*q) {
            if (spe_next_kv(&q, key, sizeof(key), &val, &hv, FALSE) && hv) {
                if (strcasecmp(key, "dc") == 0) {
                    sb_puts(&line, " dc ");
                    spe_format_expr(&line, val.buf, FALSE);
                } else if (strcasecmp(key, "mag") == 0) {
                    sb_puts(&line, " ac ");
                    spe_format_expr(&line, val.buf, FALSE);
                }
            }
        }
        sl_emit(out, line.buf);
        sb_free(&val); sb_free(&line); sb_free(&nodes);
        return;
    }

    /* Standard SPICE device or subckt */
    char pfx = 'x';
    switch (kind) {
    case SPE_KIND_MOS: pfx = 'm'; break;
    case SPE_KIND_RES: pfx = 'r'; break;
    case SPE_KIND_CAP: pfx = 'c'; break;
    case SPE_KIND_DIO: pfx = 'd'; break;
    case SPE_KIND_BJT: pfx = 'q'; break;
    case SPE_KIND_IND: pfx = 'l'; break;
    case SPE_KIND_SUBCKT:
    case SPE_KIND_UNKNOWN:
    default: pfx = 'x'; break;
    }

    SpeBuf line;
    sb_init(&line);
    /* Ensure SPICE element letter matches device type */
    if (tolower((unsigned char)inst[0]) == pfx) {
        sb_puts(&line, inst);
    } else {
        sb_putc(&line, pfx);
        sb_putc(&line, '_');
        sb_puts(&line, inst);
    }
    sb_putc(&line, ' ');
    sb_puts(&line, nodes.buf);

    if (!is_prim_res && !is_prim_cap && !is_prim_ind) {
        sb_putc(&line, ' ');
        sb_puts(&line, master);
    }

    /* Emit key=val parameters */
    char key[128];
    SpeBuf val;
    sb_init(&val);
    bool has_val = FALSE;
    const char *q = p;
    while (*q) {
        if (!spe_next_kv(&q, key, sizeof(key), &val, &has_val, FALSE) || !has_val)
            continue;
        const char *okey = key;
        if (strcasecmp(key, "trise") == 0)
            okey = "dtemp";
        /* For primitive resistor/capacitor/inductor, ngspice accepts `r={expr}` / `c={expr}` / `l={expr}` */
        sb_putc(&line, ' ');
        sb_puts(&line, okey);
        sb_putc(&line, '=');
        spe_format_expr(&line, val.buf, FALSE);
    }

    sl_emit(out, line.buf);
    sb_free(&val);
    sb_free(&line);
    sb_free(&nodes);
}

/* =========================================================================
 * Public detection & translation entry points
 * ========================================================================= */

bool inp_spectre_should_translate(FILE *fp, const char *file_name)
{
    if (!fp)
        return FALSE;

    /* 1. Explicit `.scs` extension triggers Spectre translator automatically */
    if (file_name) {
        size_t n = strlen(file_name);
        if (n >= 4 && strcasecmp(file_name + n - 4, ".scs") == 0)
            return TRUE;
    }

    /* 2. Peek first ~32 non-empty lines for `simulator lang=spectre` or Spectre keywords
     *    (when `newcompat.spe` is enabled OR file declares `simulator lang=spectre`) */
    long orig_pos = ftell(fp);
    if (orig_pos < 0)
        return FALSE;

    bool is_spe = FALSE;
    char buf[1024];
    int checked = 0;
    while (checked < 40 && fgets(buf, sizeof(buf), fp)) {
        char *p = buf;
        if (strncmp(p, "\xEF\xBB\xBF", 3) == 0) p += 3;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || *p == '*' || (p[0] == '/' && p[1] == '/'))
            continue;
        checked++;
        if (strncasecmp(p, "simulator", 9) == 0 && isspace((unsigned char)p[9])) {
            char low[256];
            size_t i;
            for (i = 0; p[i] && i + 1 < sizeof(low); i++)
                low[i] = (char)tolower((unsigned char)p[i]);
            low[i] = '\0';
            if (strstr(low, "lang=spectre")) {
                is_spe = TRUE;
                break;
            }
            if (strstr(low, "lang=spice")) {
                is_spe = FALSE;
                break;
            }
        }
        /* In explicit `set ngbehavior=spe` mode, detect Spectre structural keywords */
        if (newcompat.spe) {
            if ((strncasecmp(p, "section", 7) == 0 && isspace((unsigned char)p[7])) ||
                (strncasecmp(p, "parameters", 10) == 0 && (isspace((unsigned char)p[10]) || !p[10])) ||
                (strncasecmp(p, "inline", 6) == 0 && isspace((unsigned char)p[6])) ||
                (strncasecmp(p, "subckt", 6) == 0 && isspace((unsigned char)p[6])) ||
                (strncasecmp(p, "library", 7) == 0 && isspace((unsigned char)p[7])) ||
                (strncasecmp(p, "model", 5) == 0 && isspace((unsigned char)p[5])) ||
                (strncasecmp(p, "ahdl_include", 12) == 0 && isspace((unsigned char)p[12]))) {
                is_spe = TRUE;
                break;
            }
        }
        /* Standard SPICE dot-card at top => not Spectre unless `simulator lang=spectre` appeared */
        if (*p == '.')
            break;
    }

    fseek(fp, orig_pos, SEEK_SET);
    return is_spe;
}

char **inp_spectre_translate_fp(FILE *fp, const char *dir_name,
                                const char *file_name, bool is_top_level,
                                size_t *out_nlines)
{
    *out_nlines = 0;

    /* Ensure Spectre/HSPICE binning & parameter compatibility is active */
    newcompat.spe = TRUE;
    newcompat.isset = TRUE;

    SpeStmtList stmts;
    st_init(&stmts);
    spe_read_statements(fp, &stmts);

    /* Pass 1: collect global symbol table (models, subckts, ahdl_include VA modules) */
    SpeSymTab syms;
    sym_init(&syms);

    bool in_sub_scan = FALSE;
    for (size_t i = 0; i < stmts.count; i++) {
        if (stmts.is_spice[i])
            continue;
        const char *s = stmts.stmts[i];
        while (*s && isspace((unsigned char)*s)) s++;

        if (strncasecmp(s, "ahdl_include", 12) == 0 && isspace((unsigned char)s[12])) {
            const char *q = s + 12;
            while (*q && (isspace((unsigned char)*q) || *q == '"' || *q == '\'')) q++;
            char va_file[1024];
            size_t k = 0;
            while (*q && !isspace((unsigned char)*q) && *q != '"' && *q != '\'' && k + 1 < sizeof(va_file))
                va_file[k++] = *q++;
            va_file[k] = '\0';
            if (va_file[0])
                spe_handle_ahdl_include(va_file, dir_name, &syms);
            continue;
        }
        if ((strncasecmp(s, "inline", 6) == 0 && isspace((unsigned char)s[6])) ||
            (strncasecmp(s, "subckt", 6) == 0 && isspace((unsigned char)s[6]))) {
            const char *q = s;
            if (strncasecmp(q, "inline", 6) == 0) {
                q += 6;
                while (*q && isspace((unsigned char)*q)) q++;
            }
            if (strncasecmp(q, "subckt", 6) == 0 && isspace((unsigned char)q[6])) {
                q += 6;
                while (*q && isspace((unsigned char)*q)) q++;
                char sname[256];
                size_t k = 0;
                while (*q && !isspace((unsigned char)*q) && *q != '(' && k + 1 < sizeof(sname))
                    sname[k++] = *q++;
                sname[k] = '\0';
                sym_add(&syms, sname, SPE_KIND_SUBCKT, 0);
                in_sub_scan = TRUE;
            }
            continue;
        }
        if (strncasecmp(s, "ends", 4) == 0 && (!s[4] || isspace((unsigned char)s[4]))) {
            in_sub_scan = FALSE;
            continue;
        }
        if (!in_sub_scan && strncasecmp(s, "model", 5) == 0 && isspace((unsigned char)s[5])) {
            const char *q = s + 5;
            while (*q && isspace((unsigned char)*q)) q++;
            char mname[256], mtype[128];
            size_t k = 0;
            while (*q && !isspace((unsigned char)*q) && k + 1 < sizeof(mname)) mname[k++] = *q++;
            mname[k] = '\0';
            while (*q && isspace((unsigned char)*q)) q++;
            k = 0;
            while (*q && !isspace((unsigned char)*q) && *q != '{' && k + 1 < sizeof(mtype)) mtype[k++] = *q++;
            mtype[k] = '\0';
            SpeKind kd = spectre_model_type_to_kind(mtype);
            if (kd != SPE_KIND_UNKNOWN)
                sym_add(&syms, mname, kd, 0);
        }
    }

    /* Pass 2: emit translated SPICE lines */
    SpeLines out;
    sl_init(&out);

    /* Top-level SPICE decks consume line 0 as title; prepend a synthetic title line */
    if (is_top_level) {
        SpeBuf title; sb_init(&title);
        sb_puts(&title, "* Spectre deck: ");
        sb_puts(&title, file_name ? file_name : "(stdin)");
        sl_emit(&out, title.buf);
        sb_free(&title);
    }

    for (size_t i = 0; i < stmts.count; i++) {
        if (stmts.is_spice[i]) {
            sl_emit(&out, stmts.stmts[i]);
            continue;
        }
        const char *s = stmts.stmts[i];
        while (*s && isspace((unsigned char)*s)) s++;
        if (!*s)
            continue;

        /* Skip library / endlibrary / statistics / ahdl_include / options */
        if ((strncasecmp(s, "library", 7) == 0 && (!s[7] || isspace((unsigned char)s[7]))) ||
            (strncasecmp(s, "endlibrary", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]))) ||
            (strncasecmp(s, "statistics", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]) || s[10] == '{')) ||
            (strncasecmp(s, "ahdl_include", 12) == 0 && isspace((unsigned char)s[12]))) {
            continue;
        }

        /* section <name> -> .lib <name> */
        if (strncasecmp(s, "section", 7) == 0 && isspace((unsigned char)s[7])) {
            const char *q = s + 7;
            while (*q && isspace((unsigned char)*q)) q++;
            SpeBuf line; sb_init(&line);
            sb_puts(&line, ".lib ");
            sb_puts(&line, q);
            sl_emit(&out, line.buf);
            sb_free(&line);
            continue;
        }

        /* endsection [name] -> .endl [name] */
        if (strncasecmp(s, "endsection", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]))) {
            const char *q = s + 10;
            while (*q && isspace((unsigned char)*q)) q++;
            SpeBuf line; sb_init(&line);
            sb_puts(&line, ".endl");
            if (*q) {
                sb_putc(&line, ' ');
                sb_puts(&line, q);
            }
            sl_emit(&out, line.buf);
            sb_free(&line);
            continue;
        }

        /* include "file" [section=sec] -> .lib "file" sec | .include "file" */
        if (strncasecmp(s, "include", 7) == 0 && isspace((unsigned char)s[7])) {
            const char *q = s + 7;
            while (*q && isspace((unsigned char)*q)) q++;
            char fpath[1024];
            size_t k = 0;
            if (*q == '"' || *q == '\'') {
                char quote = *q++;
                while (*q && *q != quote && k + 1 < sizeof(fpath)) fpath[k++] = *q++;
                if (*q == quote) q++;
            } else {
                while (*q && !isspace((unsigned char)*q) && k + 1 < sizeof(fpath)) fpath[k++] = *q++;
            }
            fpath[k] = '\0';

            char sec[256] = "";
            const char *sp = strstr(q, "section=");
            if (!sp) sp = strstr(q, "SECTION=");
            if (sp) {
                sp += 8;
                while (*sp && (*sp == '"' || *sp == '\'')) sp++;
                k = 0;
                while (*sp && !isspace((unsigned char)*sp) && *sp != '"' && *sp != '\'' && k + 1 < sizeof(sec))
                    sec[k++] = *sp++;
                sec[k] = '\0';
            }

            SpeBuf line; sb_init(&line);
            if (sec[0]) {
                sb_puts(&line, ".lib \"");
                sb_puts(&line, fpath);
                sb_puts(&line, "\" ");
                sb_puts(&line, sec);
            } else {
                sb_puts(&line, ".include \"");
                sb_puts(&line, fpath);
                sb_putc(&line, '"');
            }
            sl_emit(&out, line.buf);
            sb_free(&line);
            continue;
        }

        /* Global / section-level `parameters ...` -> `.param ...` */
        if (strncasecmp(s, "parameters", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]))) {
            const char *q = s + 10;
            char key[128];
            SpeBuf val; sb_init(&val);
            bool hv = FALSE;
            while (*q) {
                if (!spe_next_kv(&q, key, sizeof(key), &val, &hv, FALSE) || !hv)
                    continue;
                SpeBuf line; sb_init(&line);
                sb_puts(&line, ".param ");
                sb_puts(&line, key);
                sb_putc(&line, '=');
                spe_format_expr(&line, val.buf, FALSE);
                sl_emit(&out, line.buf);
                sb_free(&line);
            }
            sb_free(&val);
            continue;
        }

        /* Global / section-level `model ...` */
        if (strncasecmp(s, "model", 5) == 0 && isspace((unsigned char)s[5])) {
            spe_translate_model(s, &out);
            continue;
        }

        /* [inline] subckt name [(ports...)] ... ends [name] */
        if ((strncasecmp(s, "inline", 6) == 0 && isspace((unsigned char)s[6])) ||
            (strncasecmp(s, "subckt", 6) == 0 && isspace((unsigned char)s[6]))) {
            const char *q = s;
            if (strncasecmp(q, "inline", 6) == 0) {
                q += 6;
                while (*q && isspace((unsigned char)*q)) q++;
            }
            if (strncasecmp(q, "subckt", 6) == 0 && isspace((unsigned char)q[6])) {
                q += 6;
                while (*q && isspace((unsigned char)*q)) q++;
                char sname[256];
                size_t k = 0;
                while (*q && !isspace((unsigned char)*q) && *q != '(' && k + 1 < sizeof(sname))
                    sname[k++] = *q++;
                sname[k] = '\0';

                /* Collect ports (either inside `( p1 p2 ... )` or bare tokens) */
                SpeBuf ports; sb_init(&ports);
                while (*q && isspace((unsigned char)*q)) q++;
                if (*q == '(') {
                    q++;
                    while (*q && *q != ')') {
                        while (*q && isspace((unsigned char)*q)) q++;
                        if (!*q || *q == ')') break;
                        if (ports.len > 0) sb_putc(&ports, ' ');
                        while (*q && !isspace((unsigned char)*q) && *q != ')')
                            sb_putc(&ports, *q++);
                    }
                } else {
                    while (*q) {
                        while (*q && isspace((unsigned char)*q)) q++;
                        if (!*q) break;
                        if (ports.len > 0) sb_putc(&ports, ' ');
                        while (*q && !isspace((unsigned char)*q))
                            sb_putc(&ports, *q++);
                    }
                }

                /* Find matching `ends` statement index `end_i` and scan local models */
                size_t end_i = i + 1;
                sym_clear_local(&syms);
                while (end_i < stmts.count) {
                    if (!stmts.is_spice[end_i]) {
                        const char *es = stmts.stmts[end_i];
                        while (*es && isspace((unsigned char)*es)) es++;
                        if (strncasecmp(es, "ends", 4) == 0 && (!es[4] || isspace((unsigned char)es[4])))
                            break;
                        if (strncasecmp(es, "model", 5) == 0 && isspace((unsigned char)es[5])) {
                            const char *mq = es + 5;
                            while (*mq && isspace((unsigned char)*mq)) mq++;
                            char mn[256], mt[128];
                            size_t mk = 0;
                            while (*mq && !isspace((unsigned char)*mq) && mk + 1 < sizeof(mn)) mn[mk++] = *mq++;
                            mn[mk] = '\0';
                            while (*mq && isspace((unsigned char)*mq)) mq++;
                            mk = 0;
                            while (*mq && !isspace((unsigned char)*mq) && *mq != '{' && mk + 1 < sizeof(mt)) mt[mk++] = *mq++;
                            mt[mk] = '\0';
                            SpeKind kd = spectre_model_type_to_kind(mt);
                            if (kd != SPE_KIND_UNKNOWN)
                                sym_add(&syms, mn, kd, 1);
                        }
                    }
                    end_i++;
                }

                /* Build `.subckt sname ports [formal_params...]` and local `.param` lines
                 * from all `parameters` statements inside the subckt */
                SpeBuf sub_hdr; sb_init(&sub_hdr);
                sb_puts(&sub_hdr, ".subckt ");
                sb_puts(&sub_hdr, sname);
                if (ports.len > 0) {
                    sb_putc(&sub_hdr, ' ');
                    sb_puts(&sub_hdr, ports.buf);
                }

                SpeLines local_params;
                sl_init(&local_params);

                for (size_t j = i + 1; j < end_i; j++) {
                    if (stmts.is_spice[j]) continue;
                    const char *ps = stmts.stmts[j];
                    while (*ps && isspace((unsigned char)*ps)) ps++;
                    if (strncasecmp(ps, "parameters", 10) != 0 ||
                        (ps[10] && !isspace((unsigned char)ps[10])))
                        continue;

                    const char *pq = ps + 10;
                    char key[128];
                    SpeBuf val; sb_init(&val);
                    bool hv = FALSE;
                    while (*pq) {
                        if (!spe_next_kv(&pq, key, sizeof(key), &val, &hv, TRUE))
                            continue;
                        /* Spectre reserves `m` as built-in subckt multiplicity;
                         * omit formal `m=` so ngspice applies caller `m=` automatically */
                        if (strcasecmp(key, "m") == 0)
                            continue;
                        if (!hv) {
                            /* Bare formal parameter (e.g., `parameters l w`) -> default 0 */
                            sb_putc(&sub_hdr, ' ');
                            sb_puts(&sub_hdr, key);
                            sb_puts(&sub_hdr, "=0");
                        } else if (is_plain_number(val.buf) || is_simple_val(val.buf)) {
                            /* Literal or simple identifier default -> formal subckt param */
                            sb_putc(&sub_hdr, ' ');
                            sb_puts(&sub_hdr, key);
                            sb_putc(&sub_hdr, '=');
                            sb_puts(&sub_hdr, val.buf);
                        } else {
                            /* Compound expression -> local `.param` inside subckt */
                            SpeBuf pl; sb_init(&pl);
                            sb_puts(&pl, ".param ");
                            sb_puts(&pl, key);
                            sb_putc(&pl, '=');
                            spe_format_expr(&pl, val.buf, FALSE);
                            sl_emit(&local_params, pl.buf);
                            sb_free(&pl);
                        }
                    }
                    sb_free(&val);
                }

                sl_emit(&out, sub_hdr.buf);
                sb_free(&sub_hdr);
                sb_free(&ports);

                /* Emit local `.param` lines */
                for (size_t lp_i = 0; lp_i < local_params.count; lp_i++)
                    sl_push_owned(&out, local_params.lines[lp_i]);
                tfree(local_params.lines);

                /* Emit subckt body (models and instances) */
                for (size_t j = i + 1; j < end_i; j++) {
                    if (stmts.is_spice[j]) {
                        sl_emit(&out, stmts.stmts[j]);
                        continue;
                    }
                    const char *bs = stmts.stmts[j];
                    while (*bs && isspace((unsigned char)*bs)) bs++;
                    if (!*bs) continue;
                    if (strncasecmp(bs, "parameters", 10) == 0 && (!bs[10] || isspace((unsigned char)bs[10])))
                        continue;
                    if (strncasecmp(bs, "model", 5) == 0 && isspace((unsigned char)bs[5])) {
                        spe_translate_model(bs, &out);
                        continue;
                    }
                    spe_translate_instance(bs, &syms, &out);
                }

                SpeBuf end_l; sb_init(&end_l);
                sb_puts(&end_l, ".ends ");
                sb_puts(&end_l, sname);
                sl_emit(&out, end_l.buf);
                sb_free(&end_l);

                sym_clear_local(&syms);
                i = end_i; /* advance past `ends` */
                continue;
            }
        }

        /* global <nodes...> -> .global <non-zero nodes...> */
        if (strncasecmp(s, "global", 6) == 0 && isspace((unsigned char)s[6])) {
            const char *q = s + 6;
            SpeBuf line; sb_init(&line);
            bool any = FALSE;
            while (*q) {
                while (*q && isspace((unsigned char)*q)) q++;
                if (!*q) break;
                const char *st = q;
                while (*q && !isspace((unsigned char)*q)) q++;
                size_t len = (size_t)(q - st);
                if (len == 1 && st[0] == '0')
                    continue;
                if (!any) {
                    sb_puts(&line, ".global");
                    any = TRUE;
                }
                sb_putc(&line, ' ');
                sb_putn(&line, st, len);
            }
            if (any)
                sl_emit(&out, line.buf);
            sb_free(&line);
            continue;
        }

        /* Skip standalone Spectre `options` statements inside model decks */
        if (strstr(s, " options ") || (strncasecmp(s, "options", 7) == 0 && isspace((unsigned char)s[7])))
            continue;

        /* Top-level Spectre device instance */
        if (strchr(s, '(') && strchr(s, ')')) {
            spe_translate_instance(s, &syms, &out);
            continue;
        }

        /* Top-level Spectre analysis statements: `<name> dc|tran|ac ...` */
        if (is_top_level) {
            const char *q = s;
            while (*q && !isspace((unsigned char)*q)) q++; /* skip analysis instance name */
            while (*q && isspace((unsigned char)*q)) q++;
            if (strncasecmp(q, "dc", 2) == 0 && (!q[2] || isspace((unsigned char)q[2]))) {
                sl_emit(&out, ".op");
            } else if (strncasecmp(q, "tran", 4) == 0 && (!q[4] || isspace((unsigned char)q[4]))) {
                const char *aq = q + 4;
                char key[128]; SpeBuf val; sb_init(&val); bool hv = FALSE;
                SpeBuf stop_v, step_v; sb_init(&stop_v); sb_init(&step_v);
                while (*aq) {
                    if (spe_next_kv(&aq, key, sizeof(key), &val, &hv, FALSE) && hv) {
                        if (strcasecmp(key, "stop") == 0) { stop_v.len = 0; sb_puts(&stop_v, val.buf); }
                        else if (strcasecmp(key, "step") == 0 || strcasecmp(key, "maxstep") == 0) {
                            if (step_v.len == 0) sb_puts(&step_v, val.buf);
                        }
                    }
                }
                if (stop_v.len > 0) {
                    SpeBuf line; sb_init(&line);
                    sb_puts(&line, ".tran ");
                    if (step_v.len > 0)
                        spe_format_expr(&line, step_v.buf, FALSE);
                    else {
                        sb_puts(&line, "{(");
                        sb_puts(&line, stop_v.buf);
                        sb_puts(&line, ")/100}");
                    }
                    sb_putc(&line, ' ');
                    spe_format_expr(&line, stop_v.buf, FALSE);
                    sl_emit(&out, line.buf);
                    sb_free(&line);
                }
                sb_free(&stop_v); sb_free(&step_v); sb_free(&val);
            } else if (strncasecmp(q, "ac", 2) == 0 && (!q[2] || isspace((unsigned char)q[2]))) {
                const char *aq = q + 2;
                char key[128]; SpeBuf val; sb_init(&val); bool hv = FALSE;
                SpeBuf start_v, stop_v, dec_v;
                sb_init(&start_v); sb_init(&stop_v); sb_init(&dec_v);
                while (*aq) {
                    if (spe_next_kv(&aq, key, sizeof(key), &val, &hv, FALSE) && hv) {
                        if (strcasecmp(key, "start") == 0) { start_v.len = 0; sb_puts(&start_v, val.buf); }
                        else if (strcasecmp(key, "stop") == 0) { stop_v.len = 0; sb_puts(&stop_v, val.buf); }
                        else if (strcasecmp(key, "dec") == 0) { dec_v.len = 0; sb_puts(&dec_v, val.buf); }
                    }
                }
                if (start_v.len > 0 && stop_v.len > 0) {
                    SpeBuf line; sb_init(&line);
                    sb_puts(&line, ".ac dec ");
                    sb_puts(&line, dec_v.len > 0 ? dec_v.buf : "10");
                    sb_putc(&line, ' ');
                    spe_format_expr(&line, start_v.buf, FALSE);
                    sb_putc(&line, ' ');
                    spe_format_expr(&line, stop_v.buf, FALSE);
                    sl_emit(&out, line.buf);
                    sb_free(&line);
                }
                sb_free(&start_v); sb_free(&stop_v); sb_free(&dec_v); sb_free(&val);
            }
        }
    }

    if (is_top_level)
        sl_emit(&out, ".end");

    sym_free(&syms);
    st_free(&stmts);

    *out_nlines = out.count;
    return out.lines;
}
