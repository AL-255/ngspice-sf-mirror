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
    SPE_KIND_OSDI,       /* ahdl_include VA module (synthesizes __spe_va_<uid> model) */
    SPE_KIND_OSDI_MODEL  /* named CMC OSDI model (e.g., bsimimg, juncap200, psp103) */
} SpeKind;

typedef struct {
    char   *name;
    char   *osdi_type;   /* canonical OSDI module name when kind == SPE_KIND_OSDI_MODEL */
    SpeKind kind;
    int     scope;       /* 0 = global/file, 1 = local to current subckt */
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
    for (size_t i = 0; i < t->count; i++) {
        tfree(t->items[i].name);
        tfree(t->items[i].osdi_type);
    }
    tfree(t->items);
    t->count = t->cap = 0;
}

static void sym_add_ex(SpeSymTab *t, const char *name, SpeKind kind,
                       const char *osdi_type, int scope)
{
    if (!name || !*name)
        return;
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].scope == scope && strcasecmp(t->items[i].name, name) == 0) {
            t->items[i].kind = kind;
            tfree(t->items[i].osdi_type);
            t->items[i].osdi_type = osdi_type ? copy(osdi_type) : NULL;
            return;
        }
    }
    if (t->count >= t->cap) {
        t->cap = t->cap ? t->cap * 2 : 128;
        t->items = TREALLOC(SpeSym, t->items, t->cap);
    }
    t->items[t->count].name = copy(name);
    t->items[t->count].osdi_type = osdi_type ? copy(osdi_type) : NULL;
    t->items[t->count].kind = kind;
    t->items[t->count].scope = scope;
    t->count++;
}

static void sym_add(SpeSymTab *t, const char *name, SpeKind kind, int scope)
{
    sym_add_ex(t, name, kind, NULL, scope);
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
            tfree(t->items[i].osdi_type);
        }
    }
    t->count = out;
}

/* Persistent cross-file model registry so subckts referencing models defined in
 * separately included files (e.g., `inline subckt nfet` -> `include "./model_nfet.scs"`)
 * resolve the exact device kind and OSDI module type. */
static SpeSymTab spe_global_models = { NULL, 0, 0 };

static void spe_register_global_model(const char *name, SpeKind kind, const char *osdi_type)
{
    if (!name || !*name || kind == SPE_KIND_UNKNOWN || kind == SPE_KIND_SUBCKT)
        return;
    sym_add_ex(&spe_global_models, name, kind, osdi_type, 0);
}

static const SpeSym *sym_lookup_full(const SpeSymTab *t, const char *name, bool prefer_model)
{
    /* 1. Local scope (e.g., local BJT/diode model sharing name with enclosing subckt) */
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].scope == 1 && strcasecmp(t->items[i].name, name) == 0)
            return &t->items[i];
    }
    /* 2. When inside an inline subckt whose inner instance has the same name as the
     *    subckt itself, always resolve to a model (never self-recursive subckt) */
    if (prefer_model) {
        for (size_t i = 0; i < t->count; i++) {
            if (t->items[i].scope == 0 && t->items[i].kind != SPE_KIND_SUBCKT &&
                strcasecmp(t->items[i].name, name) == 0)
                return &t->items[i];
        }
        for (size_t i = 0; i < spe_global_models.count; i++) {
            if (strcasecmp(spe_global_models.items[i].name, name) == 0)
                return &spe_global_models.items[i];
        }
        return NULL;
    }
    /* 3. File scope */
    for (size_t i = 0; i < t->count; i++) {
        if (t->items[i].scope == 0 && strcasecmp(t->items[i].name, name) == 0)
            return &t->items[i];
    }
    /* 4. Cross-file global model registry */
    for (size_t i = 0; i < spe_global_models.count; i++) {
        if (strcasecmp(spe_global_models.items[i].name, name) == 0)
            return &spe_global_models.items[i];
    }
    return NULL;
}

static SpeKind sym_lookup(const SpeSymTab *t, const char *name)
{
    const SpeSym *s = sym_lookup_full(t, name, FALSE);
    return s ? s->kind : SPE_KIND_UNKNOWN;
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

/* Return number of terminals declared by a loaded OSDI device, or -1 if not found. */
static int spe_osdi_num_terms(const char *name)
{
#ifdef OSDI
    if (!ft_sim || !name)
        return -1;
    for (int t = 0; t < ft_sim->numDevices; t++) {
        IFdevice *dev = ft_sim->devices[t];
        if (dev && dev->registry_entry && dev->name &&
            strcasecmp(dev->name, name) == 0 && dev->terms)
            return *(dev->terms);
    }
#else
    (void)name;
#endif
    return -1;
}

/* Return canonical OSDI module name if `mtype` is a CMC/industry compact model
 * supported via OSDI (rather than built-in SPICE primitives), else NULL. */
static const char *spe_cmc_osdi_type(const char *mtype)
{
    if (!mtype) return NULL;
    if (strcasecmp(mtype, "bsimimg") == 0)   return "bsimimg";
    if (strcasecmp(mtype, "juncap200") == 0) return "juncap200";
    if (strcasecmp(mtype, "psp103") == 0 || strcasecmp(mtype, "psp102") == 0 ||
        strcasecmp(mtype, "psp104") == 0)    return "psp103";
    if (strcasecmp(mtype, "bsimcmg") == 0)   return "bsimcmg";
    if (strcasecmp(mtype, "bsimbulk") == 0 || strcasecmp(mtype, "bsim6") == 0)
        return "bsimbulk";
    if (strcasecmp(mtype, "r3_cmc") == 0)    return "r3_cmc";
    if (strcasecmp(mtype, "r2_cmc") == 0)    return "r2_cmc";
    if (strcasecmp(mtype, "hicum2") == 0)    return "hicum2";
    if (strcasecmp(mtype, "mextram") == 0)   return "mextram";
    return NULL;
}

/* Attempt to auto-load `<stem>.osdi` from `<dir_name>`, `$NGSPICE_OSDI_DIR`, or
 * `$HOME/.cache/ngspice/osdi`. Safe to call repeatedly (deduplicated). */
static void spe_try_load_osdi_by_stem(const char *stem, const char *dir_name)
{
#ifdef OSDI
    if (!stem || !*stem)
        return;
    if (spe_is_loaded_osdi_device(stem))
        return;

    SpeBuf cand;
    sb_init(&cand);

    const char *dirs[3];
    int ndirs = 0;
    if (dir_name && *dir_name)
        dirs[ndirs++] = dir_name;
    const char *odir = getenv("NGSPICE_OSDI_DIR");
    if (odir && *odir)
        dirs[ndirs++] = odir;

    SpeBuf home_cache;
    sb_init(&home_cache);
    const char *home = getenv("HOME");
    if (home && *home) {
        sb_puts(&home_cache, home);
        sb_puts(&home_cache, "/.cache/ngspice/osdi");
        dirs[ndirs++] = home_cache.buf;
    }

    for (int i = 0; i < ndirs; i++) {
        cand.len = 0; cand.buf[0] = '\0';
        sb_puts(&cand, dirs[i]);
        sb_putc(&cand, '/');
        sb_puts(&cand, stem);
        sb_puts(&cand, ".osdi");
        if (spe_file_exists(cand.buf)) {
            if (!spe_osdi_already_loaded(cand.buf)) {
                if (load_osdi(cand.buf) == 0)
                    spe_mark_osdi_loaded(cand.buf);
                else
                    fprintf(stderr, "Warning: Spectre failed to load OSDI '%s'\n", cand.buf);
            }
            sb_free(&home_cache);
            sb_free(&cand);
            return;
        }
    }

    if (!spe_va_already_warned(stem)) {
        fprintf(stderr,
                "Warning: Spectre model type '%s': no compiled '%s.osdi' found "
                "(place '%s.osdi' in $NGSPICE_OSDI_DIR or ~/.cache/ngspice/osdi)\n",
                stem, stem, stem);
    }
    sb_free(&home_cache);
    sb_free(&cand);
#else
    (void)stem;
    (void)dir_name;
#endif
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
        if (*p == '+' && clean.len > 0 && clean.buf[clean.len - 1] == '+') {
            /* Collapse redundant unary `++` (e.g. `a + +b`) -> `+` for numparam */
            p++;
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

/* Return TRUE if `expr` contains a circuit-time node voltage `v(...)` or
 * branch current `i(...)` reference (for behavioral R/C/B formatting). */
static bool spe_expr_has_vi_ref(const char *expr)
{
    for (const char *p = expr; *p; p++) {
        if ((*p == 'v' || *p == 'V' || *p == 'i' || *p == 'I') &&
            (p == expr || !is_ident_char(p[-1]))) {
            const char *q = p + 1;
            while (*q && isspace((unsigned char)*q)) q++;
            if (*q == '(')
                return TRUE;
        }
    }
    return FALSE;
}

/* Format a behavioral expression containing v(...) or i(...) for ngspice R/C/B:
 * replaces standalone `temp` with `temper`, removes outer whitespace, wraps in '...'
 * when `wrap_quotes` is TRUE. */
static void spe_format_beh_expr(SpeBuf *out, const char *expr, bool wrap_quotes)
{
    SpeBuf clean;
    sb_init(&clean);
    const char *p = expr;
    while (*p) {
        if (isspace((unsigned char)*p)) { p++; continue; }
        if (is_ident_start(*p)) {
            const char *start = p;
            while (*p && is_ident_char(*p)) p++;
            size_t len = (size_t)(p - start);
            if (len == 4 && strncasecmp(start, "temp", 4) == 0 && *p != '(')
                sb_puts(&clean, "temper");
            else
                sb_putn(&clean, start, len);
            continue;
        }
        if (*p == '+' && clean.len > 0 && clean.buf[clean.len - 1] == '+') {
            /* Collapse redundant unary `++` (e.g. `a + +b`) -> `+` for numparam */
            p++;
            continue;
        }
        sb_putc(&clean, *p++);
    }
    if (wrap_quotes && clean.len > 0 && clean.buf[0] != '\'') {
        sb_putc(out, '\'');
        sb_puts(out, clean.buf);
        sb_putc(out, '\'');
    } else {
        sb_puts(out, clean.buf);
    }
    sb_free(&clean);
}

static SpeKind spectre_model_type_to_kind(const char *mtype)
{
    if (spe_cmc_osdi_type(mtype) != NULL)
        return SPE_KIND_OSDI_MODEL;
    if (strcasecmp(mtype, "bsim4") == 0 || strcasecmp(mtype, "bsim3v3") == 0 ||
        strcasecmp(mtype, "bsim3") == 0 || strcasecmp(mtype, "mos1") == 0 ||
        strcasecmp(mtype, "mos2") == 0 || strcasecmp(mtype, "mos3") == 0 ||
        strcasecmp(mtype, "level1") == 0 || strcasecmp(mtype, "ekv") == 0 ||
        strcasecmp(mtype, "bsim4v5") == 0 || strcasecmp(mtype, "hisim_hv") == 0 ||
        strcasecmp(mtype, "hisimhv") == 0 || strcasecmp(mtype, "hisimhv2") == 0)
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
 * Lightweight cross-file model prescanner (populates spe_global_models)
 * ========================================================================= */

static char **spe_prescanned_files = NULL;
static size_t spe_num_prescanned = 0, spe_cap_prescanned = 0;

static bool spe_already_prescanned(const char *path)
{
    for (size_t i = 0; i < spe_num_prescanned; i++) {
        if (strcmp(spe_prescanned_files[i], path) == 0)
            return TRUE;
    }
    if (spe_num_prescanned >= spe_cap_prescanned) {
        spe_cap_prescanned = spe_cap_prescanned ? spe_cap_prescanned * 2 : 128;
        spe_prescanned_files = TREALLOC(char *, spe_prescanned_files, spe_cap_prescanned);
    }
    spe_prescanned_files[spe_num_prescanned++] = copy(path);
    return FALSE;
}

static void spe_prescan_models_in_file(const char *fpath, const char *dir_name, int depth)
{
    if (!fpath || !*fpath || depth > 4)
        return;

    SpeBuf full;
    sb_init(&full);
    if (fpath[0] == '/' || !dir_name || !*dir_name) {
        sb_puts(&full, fpath);
    } else {
        sb_puts(&full, dir_name);
        sb_putc(&full, '/');
        sb_puts(&full, fpath);
    }

    if (spe_already_prescanned(full.buf)) {
        sb_free(&full);
        return;
    }

    FILE *f = fopen(full.buf, "r");
    if (!f) {
        sb_free(&full);
        return;
    }

    /* Compute directory of `full` for nested relative includes */
    SpeBuf fdir;
    sb_init(&fdir);
    const char *sl = strrchr(full.buf, '/');
    if (sl)
        sb_putn(&fdir, full.buf, (size_t)(sl - full.buf));
    else if (dir_name)
        sb_puts(&fdir, dir_name);

    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        if (strncmp(p, "\xEF\xBB\xBF", 3) == 0) p += 3;
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p || *p == '*' || (p[0] == '/' && p[1] == '/'))
            continue;

        if (strncasecmp(p, "model", 5) == 0 && isspace((unsigned char)p[5])) {
            const char *q = p + 5;
            while (*q && isspace((unsigned char)*q)) q++;
            char mn[256], mt[128];
            size_t k = 0;
            while (*q && !isspace((unsigned char)*q) && k + 1 < sizeof(mn)) mn[k++] = *q++;
            mn[k] = '\0';
            while (*q && isspace((unsigned char)*q)) q++;
            k = 0;
            while (*q && !isspace((unsigned char)*q) && *q != '{' && k + 1 < sizeof(mt)) mt[k++] = *q++;
            mt[k] = '\0';
            SpeKind kd = spectre_model_type_to_kind(mt);
            if (kd != SPE_KIND_UNKNOWN)
                spe_register_global_model(mn, kd, spe_cmc_osdi_type(mt));
        } else if (depth < 4 && strncasecmp(p, "include", 7) == 0 && isspace((unsigned char)p[7])) {
            const char *q = p + 7;
            while (*q && isspace((unsigned char)*q)) q++;
            char incf[1024];
            size_t k = 0;
            if (*q == '"' || *q == '\'') {
                char qt = *q++;
                while (*q && *q != qt && k + 1 < sizeof(incf)) incf[k++] = *q++;
            } else {
                while (*q && !isspace((unsigned char)*q) && k + 1 < sizeof(incf)) incf[k++] = *q++;
            }
            incf[k] = '\0';
            if (incf[0])
                spe_prescan_models_in_file(incf, fdir.buf, depth + 1);
        }
    }

    fclose(f);
    sb_free(&fdir);
    sb_free(&full);
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

/* Check if statement opens a brace-joined block (`model`, `real`, `statistics`, etc.)
 * as opposed to structural control-flow `if`/`else`. */
static bool spe_is_joined_brace_stmt(const char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    if (strncasecmp(s, "model", 5) == 0 && isspace((unsigned char)s[5])) return TRUE;
    if (strncasecmp(s, "real", 4) == 0 && isspace((unsigned char)s[4])) return TRUE;
    if (strncasecmp(s, "statistics", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]) || s[10] == '{')) return TRUE;
    if (strncasecmp(s, "process", 7) == 0 && (!s[7] || isspace((unsigned char)s[7]) || s[7] == '{')) return TRUE;
    if (strncasecmp(s, "mismatch", 8) == 0 && (!s[8] || isspace((unsigned char)s[8]) || s[8] == '{')) return TRUE;
    if (strncasecmp(s, "correlate", 9) == 0 && (!s[9] || isspace((unsigned char)s[9]) || s[9] == '{')) return TRUE;
    return FALSE;
}

/* Parse `if (<cond>)` or `else if (<cond>)` or `else` at `p`.
 * Returns: 1 = if, 2 = else if, 3 = else, 0 = none.
 * Advances `*rest_out` past the header and optional `{`. */
static int spe_parse_ctrl_header(const char *p, SpeBuf *cond_out, const char **rest_out, bool *saw_lbrace)
{
    *saw_lbrace = FALSE;
    cond_out->len = 0; cond_out->buf[0] = '\0';
    while (*p && isspace((unsigned char)*p)) p++;

    int kind = 0;
    if (strncasecmp(p, "if", 2) == 0 && (isspace((unsigned char)p[2]) || p[2] == '(')) {
        kind = 1;
        p += 2;
    } else if (strncasecmp(p, "else", 4) == 0 &&
               (!p[4] || isspace((unsigned char)p[4]) || p[4] == '{')) {
        p += 4;
        while (*p && isspace((unsigned char)*p)) p++;
        if (strncasecmp(p, "if", 2) == 0 && (isspace((unsigned char)p[2]) || p[2] == '(')) {
            kind = 2;
            p += 2;
        } else {
            kind = 3;
        }
    } else {
        return 0;
    }

    if (kind == 1 || kind == 2) {
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p != '(')
            return 0;
        p++;
        int depth = 1;
        const char *cstart = p;
        while (*p && depth > 0) {
            if (*p == '(') depth++;
            else if (*p == ')') depth--;
            if (depth > 0) p++;
        }
        if (depth != 0)
            return 0;
        sb_putn(cond_out, cstart, (size_t)(p - cstart));
        p++; /* skip ')' */
    }

    while (*p && isspace((unsigned char)*p)) p++;
    if (*p == '{') {
        *saw_lbrace = TRUE;
        p++;
        while (*p && isspace((unsigned char)*p)) p++;
    }
    *rest_out = p;
    return kind;
}

static void spe_flush_cur(SpeStmtList *out, SpeBuf *cur, bool spice_mode)
{
    if (cur->len > 0) {
        st_push(out, copy(cur->buf), spice_mode);
        cur->len = 0;
        cur->buf[0] = '\0';
    }
}

/* Read entire FILE into raw lines, strip // comments, join Spectre continuations (\ and +)
 * and brace-delimited data blocks (`model ... { ... }`, `real f(...) { ... }`, `statistics { ... }`),
 * while preserving structural `if`/`else if`/`else` blocks as `__SPE_IF__`/`__SPE_ELSEIF__`/
 * `__SPE_ELSE__`/`__SPE_ENDIF__` statements. */
static void spe_read_statements(FILE *fp, SpeStmtList *out)
{
    char linebuf[8192];
    bool in_spectre = TRUE;
    SpeBuf cur;
    sb_init(&cur);
    int join_brace_depth = 0;
    int ctrl_brace_depth = 0;
    bool pending_ctrl_lbrace = FALSE;
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
            spe_flush_cur(out, &cur, !in_spectre);
            join_brace_depth = 0;
            backslash_cont = FALSE;
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

        /* Inside a joined brace block (`model { ... }`, `real f() { ... }`, `statistics { ... }`) */
        if (join_brace_depth > 0) {
            if (cur.len > 0)
                sb_putc(&cur, (!is_plus_cont && !backslash_cont) ? '\n' : ' ');
            sb_puts(&cur, p);
            bool in_sq = FALSE, in_dq = FALSE;
            for (const char *q = p; *q; q++) {
                if (*q == '\'' && !in_dq) in_sq = !in_sq;
                else if (*q == '"' && !in_sq) in_dq = !in_dq;
                else if (!in_sq && !in_dq) {
                    if (*q == '{') join_brace_depth++;
                    else if (*q == '}' && join_brace_depth > 0) join_brace_depth--;
                }
            }
            backslash_cont = ends_backslash;
            if (join_brace_depth == 0 && !backslash_cont)
                spe_flush_cur(out, &cur, FALSE);
            continue;
        }

        /* Handle continuation of a normal (non-brace) statement */
        if (backslash_cont || is_plus_cont) {
            if (cur.len > 0) sb_putc(&cur, ' ');
            sb_puts(&cur, p);
            backslash_cont = ends_backslash;
            continue;
        }

        /* Handle standalone `{` on the line after `if (...)` / `else` */
        if (pending_ctrl_lbrace && *p == '{') {
            pending_ctrl_lbrace = FALSE;
            ctrl_brace_depth++;
            p++;
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) continue;
        } else {
            pending_ctrl_lbrace = FALSE;
        }

        /* Handle leading `}` closing a structural `if`/`else` block (incl. `} else {` / `}else {`) */
        if (*p == '}' && ctrl_brace_depth > 0) {
            spe_flush_cur(out, &cur, FALSE);
            while (*p == '}' && ctrl_brace_depth > 0) {
                ctrl_brace_depth--;
                p++;
                while (*p && isspace((unsigned char)*p)) p++;
                SpeBuf cond; sb_init(&cond);
                const char *rest = NULL;
                bool saw_lb = FALSE;
                int ck = spe_parse_ctrl_header(p, &cond, &rest, &saw_lb);
                if (ck == 2 || ck == 3) {
                    SpeBuf hdr; sb_init(&hdr);
                    if (ck == 2) { sb_puts(&hdr, "__SPE_ELSEIF__ "); sb_puts(&hdr, cond.buf); }
                    else         { sb_puts(&hdr, "__SPE_ELSE__"); }
                    st_push(out, copy(hdr.buf), FALSE);
                    sb_free(&hdr);
                    sb_free(&cond);
                    if (saw_lb) ctrl_brace_depth++;
                    else pending_ctrl_lbrace = TRUE;
                    p = (char *)rest;
                    break;
                }
                sb_free(&cond);
                st_push(out, copy("__SPE_ENDIF__"), FALSE);
            }
            if (!*p) {
                backslash_cont = ends_backslash;
                continue;
            }
        }

        /* Handle `if (...) {`, `else if (...) {`, `else {` (incl. `else` on line after `}`) */
        {
            SpeBuf cond; sb_init(&cond);
            const char *rest = NULL;
            bool saw_lb = FALSE;
            int ck = spe_parse_ctrl_header(p, &cond, &rest, &saw_lb);
            if (ck != 0) {
                spe_flush_cur(out, &cur, FALSE);
                if ((ck == 2 || ck == 3) && out->count > 0 &&
                    !out->is_spice[out->count - 1] &&
                    strcmp(out->stmts[out->count - 1], "__SPE_ENDIF__") == 0) {
                    /* Replace immediately preceding __SPE_ENDIF__ from `}\nelse` */
                    tfree(out->stmts[out->count - 1]);
                    out->count--;
                }
                SpeBuf hdr; sb_init(&hdr);
                if (ck == 1)      { sb_puts(&hdr, "__SPE_IF__ ");     sb_puts(&hdr, cond.buf); }
                else if (ck == 2) { sb_puts(&hdr, "__SPE_ELSEIF__ "); sb_puts(&hdr, cond.buf); }
                else              { sb_puts(&hdr, "__SPE_ELSE__"); }
                st_push(out, copy(hdr.buf), FALSE);
                sb_free(&hdr);
                sb_free(&cond);
                if (saw_lb) ctrl_brace_depth++;
                else pending_ctrl_lbrace = TRUE;
                p = (char *)rest;
                if (!*p) {
                    backslash_cont = ends_backslash;
                    continue;
                }
            } else {
                sb_free(&cond);
            }
        }

        /* Normal statement start */
        spe_flush_cur(out, &cur, FALSE);
        sb_puts(&cur, p);

        /* Check if this starts a brace-joined block (`model`, `real`, `statistics`, etc.) */
        if (spe_is_joined_brace_stmt(cur.buf)) {
            bool in_sq = FALSE, in_dq = FALSE;
            int delta = 0;
            bool had_brace = FALSE;
            for (const char *q = p; *q; q++) {
                if (*q == '\'' && !in_dq) in_sq = !in_sq;
                else if (*q == '"' && !in_sq) in_dq = !in_dq;
                else if (!in_sq && !in_dq) {
                    if (*q == '{') { delta++; had_brace = TRUE; }
                    else if (*q == '}' && delta > 0) { delta--; had_brace = TRUE; }
                }
            }
            join_brace_depth += delta;
            backslash_cont = ends_backslash;
            if (had_brace && join_brace_depth == 0 && !backslash_cont)
                spe_flush_cur(out, &cur, FALSE);
            continue;
        }

        /* Check for trailing `}` closing an inline `if (...) { stmt }` on one line */
        if (ctrl_brace_depth > 0 && cur.len > 0 && cur.buf[cur.len - 1] == '}') {
            int trailing_rb = 0;
            while (cur.len > 0 && cur.buf[cur.len - 1] == '}' && ctrl_brace_depth > 0) {
                cur.buf[--cur.len] = '\0';
                while (cur.len > 0 && isspace((unsigned char)cur.buf[cur.len - 1]))
                    cur.buf[--cur.len] = '\0';
                ctrl_brace_depth--;
                trailing_rb++;
            }
            spe_flush_cur(out, &cur, FALSE);
            for (int ri = 0; ri < trailing_rb; ri++)
                st_push(out, copy("__SPE_ENDIF__"), FALSE);
        }

        backslash_cont = ends_backslash;
    }

    spe_flush_cur(out, &cur, !in_spectre);
    sb_free(&cur);
}

/* Translate Spectre `real fname(real a, real b) { return <expr>[;] }` -> `.func fname(a,b) {<expr>}` */
static void spe_translate_real_func(const char *stmt, SpeLines *out)
{
    const char *p = stmt + 4; /* skip "real" */
    while (*p && isspace((unsigned char)*p)) p++;

    char fname[256];
    size_t k = 0;
    while (*p && is_ident_char(*p) && k + 1 < sizeof(fname)) fname[k++] = *p++;
    fname[k] = '\0';
    if (!fname[0])
        return;

    while (*p && isspace((unsigned char)*p)) p++;
    if (*p != '(')
        return;
    p++;

    /* Extract argument names, stripping optional `real`/`integer`/`int` type prefixes */
    SpeBuf args; sb_init(&args);
    while (*p && *p != ')') {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (!*p || *p == ')') break;
        const char *tok_start = p;
        while (*p && is_ident_char(*p)) p++;
        size_t tlen = (size_t)(p - tok_start);
        if ((tlen == 4 && strncasecmp(tok_start, "real", 4) == 0) ||
            (tlen == 3 && strncasecmp(tok_start, "int", 3) == 0) ||
            (tlen == 7 && strncasecmp(tok_start, "integer", 7) == 0)) {
            while (*p && isspace((unsigned char)*p)) p++;
            tok_start = p;
            while (*p && is_ident_char(*p)) p++;
            tlen = (size_t)(p - tok_start);
        }
        if (tlen > 0) {
            if (args.len > 0) sb_putc(&args, ',');
            sb_putn(&args, tok_start, tlen);
        }
        while (*p && *p != ',' && *p != ')') p++;
    }
    if (*p == ')') p++;

    const char *lb = strchr(p, '{');
    const char *rb = strrchr(p, '}');
    if (!lb || !rb || rb <= lb) {
        sb_free(&args);
        return;
    }

    const char *body = lb + 1;
    while (body < rb && isspace((unsigned char)*body)) body++;
    if (strncasecmp(body, "return", 6) == 0 && (isspace((unsigned char)body[6]) || body[6] == '(')) {
        body += 6;
        while (body < rb && isspace((unsigned char)*body)) body++;
    }
    const char *bend = rb;
    while (bend > body && (isspace((unsigned char)bend[-1]) || bend[-1] == ';'))
        bend--;

    SpeBuf expr; sb_init(&expr);
    sb_putn(&expr, body, (size_t)(bend - body));

    SpeBuf line; sb_init(&line);
    sb_puts(&line, ".func ");
    sb_puts(&line, fname);
    sb_putc(&line, '(');
    sb_puts(&line, args.buf);
    sb_puts(&line, ") ");
    spe_format_expr(&line, expr.buf, TRUE);
    sl_emit(out, line.buf);

    sb_free(&line);
    sb_free(&expr);
    sb_free(&args);
}

/* =========================================================================
 * Model emission helper (binned and unbinned)
 * ========================================================================= */

static void spe_emit_model_params(SpeBuf *out, const char *mtype, const char *params,
                                  char *detected_polarity, size_t pol_cap,
                                  bool is_osdi_cmc, bool strip_bin_bounds)
{
    const char *p = params;
    char key[128];
    SpeBuf val;
    sb_init(&val);
    bool has_val = FALSE;
    bool is_juncap = (strcasecmp(mtype, "juncap200") == 0);
    bool is_diode  = (strcasecmp(mtype, "diode") == 0);

    while (*p) {
        if (!spe_next_kv(&p, key, sizeof(key), &val, &has_val, FALSE))
            continue;
        if (!has_val)
            continue;

        if (strcasecmp(key, "type") == 0) {
            if (detected_polarity && pol_cap > 0)
                snprintf(detected_polarity, pol_cap, "%s", val.buf);
            if (is_osdi_cmc) {
                /* CMC OSDI models (bsimimg, juncap200, psp103, ...) take integer type=1|-1 */
                sb_puts(out, " type=");
                if (strcasecmp(val.buf, "n") == 0 || strcasecmp(val.buf, "nmos") == 0)
                    sb_puts(out, "1");
                else if (strcasecmp(val.buf, "p") == 0 || strcasecmp(val.buf, "pmos") == 0)
                    sb_puts(out, "-1");
                else
                    spe_format_expr(out, val.buf, FALSE);
            }
            continue;
        }
        /* Skip Spectre-only compatibility/vector flags not recognized by SPICE/OSDI models */
        if (strcasecmp(key, "coeffs") == 0 || strcasecmp(key, "compatible") == 0)
            continue;
        if (strip_bin_bounds &&
            (strcasecmp(key, "lmin") == 0 || strcasecmp(key, "lmax") == 0 ||
             strcasecmp(key, "wmin") == 0 || strcasecmp(key, "wmax") == 0))
            continue;
        if (is_diode &&
            (strcasecmp(key, "minr") == 0 || strcasecmp(key, "imax") == 0 ||
             strcasecmp(key, "vjmin") == 0))
            continue;

        /* Map Spectre BJT `struct=vertical|lateral` -> SPICE `subs=1|-1` */
        if (strcasecmp(mtype, "bjt") == 0 && strcasecmp(key, "struct") == 0) {
            if (strcasecmp(val.buf, "vertical") == 0)
                sb_puts(out, " subs=1");
            else if (strcasecmp(val.buf, "lateral") == 0)
                sb_puts(out, " subs=-1");
            continue;
        }

        /* Map Spectre tref -> ngspice tnom (or trj for juncap200) */
        const char *out_key = key;
        if (strcasecmp(key, "tref") == 0)
            out_key = is_juncap ? "trj" : "tnom";

        /* For OSDI CMC models, filter out model params not declared by the loaded OSDI module
         * (e.g. Spectre-specific version/extension params) to avoid fatal unknown-param errors. */
        if (is_osdi_cmc) {
            const char *osdi_mod = spe_cmc_osdi_type(mtype);
            if (osdi_mod && spe_is_loaded_osdi_device(osdi_mod)) {
                if (spe_osdi_param_scope(osdi_mod, out_key) == 0)
                    continue;
            }
        }

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
    const char *cmc = spe_cmc_osdi_type(mtype);
    if (cmc) {
        snprintf(spice_type, st_cap, "%s", cmc);
        return;
    }
    bool is_p = (pol && (pol[0] == 'p' || pol[0] == 'P' || strcmp(pol, "-1") == 0));
    if (strcasecmp(mtype, "bsim4") == 0 || strcasecmp(mtype, "bsim4v5") == 0) {
        snprintf(spice_type, st_cap, "%s", is_p ? "pmos" : "nmos");
        *extra_level = " level=54";
    } else if (strcasecmp(mtype, "hisim_hv") == 0 || strcasecmp(mtype, "hisimhv") == 0 ||
               strcasecmp(mtype, "hisimhv2") == 0) {
        snprintf(spice_type, st_cap, "%s", is_p ? "pmos" : "nmos");
        *extra_level = " level=73";
    } else if (strcasecmp(mtype, "bsim3v3") == 0 || strcasecmp(mtype, "bsim3") == 0) {
        snprintf(spice_type, st_cap, "%s", is_p ? "pmos" : "nmos");
        *extra_level = " level=49";
    } else if (strcasecmp(mtype, "mos1") == 0 || strcasecmp(mtype, "level1") == 0) {
        snprintf(spice_type, st_cap, "%s", is_p ? "pmos" : "nmos");
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

static void spe_translate_model(const char *stmt, const char *dir_name, SpeLines *out)
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

    const char *cmc = spe_cmc_osdi_type(mtype);
    bool is_osdi_cmc = (cmc != NULL);
    if (is_osdi_cmc)
        spe_try_load_osdi_by_stem(cmc, dir_name);

    SpeKind mk = spectre_model_type_to_kind(mtype);
    if (mk != SPE_KIND_UNKNOWN)
        spe_register_global_model(mname, mk, cmc);

    if (*p != '{') {
        /* Unbinned model */
        char pol[32] = "";
        SpeBuf pbuf;
        sb_init(&pbuf);
        spe_emit_model_params(&pbuf, mtype, p, pol, sizeof(pol), is_osdi_cmc, is_osdi_cmc);

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

    /* First count how many bins exist (single-bin OSDI models emit without `.bin` suffix) */
    int total_bins = 0;
    {
        char *scan = body;
        while (*scan) {
            char *nl = strchr(scan, '\n');
            char *l = scan;
            while (*l && *l != '\n' && isspace((unsigned char)*l)) l++;
            char *c = l;
            while (*c && isdigit((unsigned char)*c)) c++;
            if (c > l && *c == ':')
                total_bins++;
            if (!nl) break;
            scan = nl + 1;
        }
    }
    bool omit_bin_suffix = (is_osdi_cmc && total_bins <= 1);

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
                spe_emit_model_params(&pbuf, mtype, bin_params.buf, pol, sizeof(pol),
                                      is_osdi_cmc, omit_bin_suffix);
                char stype[64];
                const char *extra = "";
                spe_resolve_spice_model_type(mtype, pol, stype, sizeof(stype), &extra);

                SpeBuf line;
                sb_init(&line);
                sb_puts(&line, ".model ");
                sb_puts(&line, mname);
                if (!omit_bin_suffix) {
                    sb_putc(&line, '.');
                    sb_puts(&line, cur_bin);
                }
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
        spe_emit_model_params(&pbuf, mtype, bin_params.buf, pol, sizeof(pol),
                              is_osdi_cmc, omit_bin_suffix);
        char stype[64];
        const char *extra = "";
        spe_resolve_spice_model_type(mtype, pol, stype, sizeof(stype), &extra);

        SpeBuf line;
        sb_init(&line);
        sb_puts(&line, ".model ");
        sb_puts(&line, mname);
        if (!omit_bin_suffix) {
            sb_putc(&line, '.');
            sb_puts(&line, cur_bin);
        }
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
 * Device instance translation (`inst [(n1 n2 ...)|n1 n2 ...] master k=v ...`)
 * ========================================================================= */

static unsigned long spe_va_uid = 0;

static bool spe_is_ignored_directive_stmt(const char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    if ((strncasecmp(s, "library", 7) == 0 && (!s[7] || isspace((unsigned char)s[7]))) ||
        (strncasecmp(s, "endlibrary", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]))) ||
        (strncasecmp(s, "statistics", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]) || s[10] == '{')) ||
        (strncasecmp(s, "process", 7) == 0 && (!s[7] || isspace((unsigned char)s[7]) || s[7] == '{')) ||
        (strncasecmp(s, "mismatch", 8) == 0 && (!s[8] || isspace((unsigned char)s[8]) || s[8] == '{')) ||
        (strncasecmp(s, "correlate", 9) == 0 && (!s[9] || isspace((unsigned char)s[9]) || s[9] == '{')) ||
        (strncasecmp(s, "ahdl_include", 12) == 0 && isspace((unsigned char)s[12])) ||
        (strncasecmp(s, "assert", 6) == 0 && isspace((unsigned char)s[6])) ||
        (strncasecmp(s, "checklimit", 10) == 0 && isspace((unsigned char)s[10])) ||
        (strncasecmp(s, "paramtest", 9) == 0 && isspace((unsigned char)s[9])) ||
        (strncasecmp(s, "options", 7) == 0 && isspace((unsigned char)s[7])))
        return TRUE;

    /* Named Spectre statements: `<name> options ...`, `<name> assert ...`, `<name> checklimit ...` */
    const char *q = s;
    while (*q && !isspace((unsigned char)*q) && *q != '(') q++;
    while (*q && isspace((unsigned char)*q)) q++;
    if ((strncasecmp(q, "options", 7) == 0 && (!q[7] || isspace((unsigned char)q[7]))) ||
        (strncasecmp(q, "assert", 6) == 0 && (!q[6] || isspace((unsigned char)q[6]))) ||
        (strncasecmp(q, "checklimit", 10) == 0 && (!q[10] || isspace((unsigned char)q[10]))) ||
        (strncasecmp(q, "paramtest", 9) == 0 && (!q[9] || isspace((unsigned char)q[9]))))
        return TRUE;

    return FALSE;
}

static void spe_translate_instance(const char *stmt, const SpeSymTab *syms,
                                   const char *cur_subckt_name, SpeLines *out)
{
    char inst[256] = "";
    char master[256] = "";
    SpeBuf nodes;
    sb_init(&nodes);
    int n_nodes = 0;
    const char *p = stmt;
    while (*p && isspace((unsigned char)*p)) p++;

    /* Check whether the instance has a parenthesized node list `inst (n1 n2 ...) master ...`
     * Note: a `(` inside an expression AFTER `key=` does NOT count as a node list! */
    const char *eq = strchr(p, '=');
    const char *lp = strchr(p, '(');
    if (lp && eq && lp > eq)
        lp = NULL;

    if (lp) {
        const char *rp = strchr(lp + 1, ')');
        if (!rp) {
            sb_free(&nodes);
            return;
        }
        size_t k = 0;
        while (p < lp && !isspace((unsigned char)*p) && k + 1 < sizeof(inst))
            inst[k++] = *p++;
        inst[k] = '\0';

        p = lp + 1;
        while (p < rp) {
            while (p < rp && isspace((unsigned char)*p)) p++;
            if (p >= rp) break;
            if (nodes.len > 0) sb_putc(&nodes, ' ');
            while (p < rp && !isspace((unsigned char)*p))
                sb_putc(&nodes, *p++);
            n_nodes++;
        }

        p = rp + 1;
        while (*p && isspace((unsigned char)*p)) p++;
        k = 0;
        while (*p && !isspace((unsigned char)*p) && k + 1 < sizeof(master))
            master[k++] = *p++;
        master[k] = '\0';
        while (*p && isspace((unsigned char)*p)) p++;
    } else {
        /* Unparenthesized Spectre instance: `inst [n1 n2 ...] master [k=v ...]` */
        const char *toks[64];
        size_t tlens[64];
        int ntoks = 0;
        const char *q = p;
        while (*q && ntoks < 64) {
            while (*q && isspace((unsigned char)*q)) q++;
            if (!*q || looks_like_kv_start(q))
                break;
            toks[ntoks] = q;
            while (*q && !isspace((unsigned char)*q)) q++;
            tlens[ntoks] = (size_t)(q - toks[ntoks]);
            ntoks++;
        }
        if (ntoks < 2) {
            SpeBuf c; sb_init(&c);
            sb_puts(&c, "* [spectre-unparsed] ");
            sb_puts(&c, stmt);
            sl_emit(out, c.buf);
            sb_free(&c);
            sb_free(&nodes);
            return;
        }
        size_t ilen = tlens[0] < sizeof(inst) - 1 ? tlens[0] : sizeof(inst) - 1;
        memcpy(inst, toks[0], ilen);
        inst[ilen] = '\0';

        size_t mlen = tlens[ntoks - 1] < sizeof(master) - 1 ? tlens[ntoks - 1] : sizeof(master) - 1;
        memcpy(master, toks[ntoks - 1], mlen);
        master[mlen] = '\0';

        for (int ti = 1; ti < ntoks - 1; ti++) {
            if (nodes.len > 0) sb_putc(&nodes, ' ');
            sb_putn(&nodes, toks[ti], tlens[ti]);
            n_nodes++;
        }
        p = q;
        while (*p && isspace((unsigned char)*p)) p++;
    }

    /* Built-in Spectre primitive check */
    bool is_prim_res  = (strcasecmp(master, "resistor") == 0);
    bool is_prim_cap  = (strcasecmp(master, "capacitor") == 0);
    bool is_prim_ind  = (strcasecmp(master, "inductor") == 0);
    bool is_prim_mut  = (strcasecmp(master, "mutual_inductor") == 0);
    bool is_prim_vsrc = (strcasecmp(master, "vsource") == 0);
    bool is_prim_isrc = (strcasecmp(master, "isource") == 0);
    bool is_prim_vcvs = (strcasecmp(master, "vcvs") == 0);
    bool is_prim_bsrc = (strcasecmp(master, "bsource") == 0);

    /* Handle `mutual_inductor`: `k1 mutual_inductor coupling=<k> ind1=<L1> ind2=<L2>` -> `k1 L1 L2 {k}` */
    if (is_prim_mut) {
        SpeBuf ind1, ind2, coup, val;
        sb_init(&ind1); sb_init(&ind2); sb_init(&coup); sb_init(&val);
        char key[128];
        bool hv = FALSE;
        const char *q = p;
        while (*q) {
            if (spe_next_kv(&q, key, sizeof(key), &val, &hv, FALSE) && hv) {
                if (strcasecmp(key, "ind1") == 0) { ind1.len = 0; sb_puts(&ind1, val.buf); }
                else if (strcasecmp(key, "ind2") == 0) { ind2.len = 0; sb_puts(&ind2, val.buf); }
                else if (strcasecmp(key, "coupling") == 0) { coup.len = 0; sb_puts(&coup, val.buf); }
            }
        }
        if (ind1.len > 0 && ind2.len > 0 && coup.len > 0) {
            SpeBuf line; sb_init(&line);
            if (tolower((unsigned char)inst[0]) == 'k') sb_puts(&line, inst);
            else { sb_puts(&line, "k_"); sb_puts(&line, inst); }
            sb_putc(&line, ' ');
            if (tolower((unsigned char)ind1.buf[0]) != 'l') sb_puts(&line, "l_");
            sb_puts(&line, ind1.buf);
            sb_putc(&line, ' ');
            if (tolower((unsigned char)ind2.buf[0]) != 'l') sb_puts(&line, "l_");
            sb_puts(&line, ind2.buf);
            sb_putc(&line, ' ');
            spe_format_expr(&line, coup.buf, FALSE);
            sl_emit(out, line.buf);
            sb_free(&line);
        }
        sb_free(&ind1); sb_free(&ind2); sb_free(&coup); sb_free(&val);
        sb_free(&nodes);
        return;
    }

    SpeKind kind = SPE_KIND_UNKNOWN;
    const char *osdi_type = NULL;
    if (is_prim_res) kind = SPE_KIND_RES;
    else if (is_prim_cap) kind = SPE_KIND_CAP;
    else if (is_prim_ind) kind = SPE_KIND_IND;
    else {
        bool prefer_model = (cur_subckt_name && strcasecmp(master, cur_subckt_name) == 0);
        const SpeSym *sym = sym_lookup_full(syms, master, prefer_model);
        if (sym) {
            kind = sym->kind;
            osdi_type = sym->osdi_type;
        } else if (spe_is_loaded_osdi_device(master)) {
            kind = SPE_KIND_OSDI;
        }
    }

    /* Handle named CMC OSDI model instantiation (`n_<inst> nodes [th] <master> k=v ...`) */
    if (kind == SPE_KIND_OSDI_MODEL) {
        bool is_juncap = (osdi_type && strcasecmp(osdi_type, "juncap200") == 0);
        SpeBuf inst_line;
        sb_init(&inst_line);
        sb_puts(&inst_line, "n_");
        sb_puts(&inst_line, inst);
        sb_putc(&inst_line, ' ');
        sb_puts(&inst_line, nodes.buf);

        /* Append isolated thermal node if OSDI module (e.g. bsimimg) declares 5 terminals
         * `(d, fg, s, bg, t)` while the Spectre instance specifies 4 electrical terminals */
        if (osdi_type) {
            int req_terms = spe_osdi_num_terms(osdi_type);
            if (req_terms == n_nodes + 1) {
                char thnode[64];
                snprintf(thnode, sizeof(thnode), " __spe_th_%lu", ++spe_va_uid);
                sb_puts(&inst_line, thnode);
            }
        }

        sb_putc(&inst_line, ' ');
        sb_puts(&inst_line, master);

        char key[128];
        SpeBuf val;
        sb_init(&val);
        bool has_val = FALSE;
        const char *q = p;
        while (*q) {
            if (!spe_next_kv(&q, key, sizeof(key), &val, &has_val, FALSE) || !has_val)
                continue;
            if (strcasecmp(key, "region") == 0 || strcasecmp(key, "isnoisy") == 0)
                continue;
            const char *okey = key;
            if (is_juncap) {
                if (strcasecmp(key, "area") == 0) okey = "ab";
                else if (strcasecmp(key, "peri") == 0 || strcasecmp(key, "perim") == 0) okey = "ls";
                else if (strcasecmp(key, "pgate") == 0) okey = "lg";
                else if (strcasecmp(key, "trise") == 0 || strcasecmp(key, "dtemp") == 0) okey = "dta";
            } else {
                if (strcasecmp(key, "trise") == 0) okey = "dtemp";
            }
            /* Filter unknown instance params when the OSDI device is loaded */
            if (osdi_type && spe_is_loaded_osdi_device(osdi_type)) {
                if (strcasecmp(okey, "m") != 0 && strcasecmp(okey, "temp") != 0 &&
                    spe_osdi_param_scope(osdi_type, okey) != 1)
                    continue;
            }
            sb_putc(&inst_line, ' ');
            sb_puts(&inst_line, okey);
            sb_putc(&inst_line, '=');
            spe_format_expr(&inst_line, val.buf, FALSE);
        }
        sl_emit(out, inst_line.buf);
        sb_free(&val);
        sb_free(&inst_line);
        sb_free(&nodes);
        return;
    }

    /* Handle ahdl_include Verilog-A module instantiation (`n_<inst> nodes __spe_va_<uid>`) */
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
            if (strcasecmp(key, "region") == 0 || strcasecmp(key, "isnoisy") == 0)
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

    /* Handle primitive vcvs / bsource / vsource / isource */
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
        sb_puts(&line, nodes.buf);
        char key[128]; SpeBuf val; sb_init(&val); bool hv = FALSE;
        const char *q = p;
        while (*q) {
            if (!spe_next_kv(&q, key, sizeof(key), &val, &hv, FALSE) || !hv)
                continue;
            sb_putc(&line, ' ');
            sb_puts(&line, key);
            sb_putc(&line, '=');
            if (strcasecmp(key, "i") == 0 || strcasecmp(key, "v") == 0)
                spe_format_beh_expr(&line, val.buf, FALSE);
            else
                spe_format_expr(&line, val.buf, FALSE);
        }
        sl_emit(out, line.buf);
        sb_free(&val); sb_free(&line); sb_free(&nodes);
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
        if (strcasecmp(key, "region") == 0 || strcasecmp(key, "isnoisy") == 0)
            continue;
        const char *okey = key;
        if (strcasecmp(key, "trise") == 0)
            okey = "dtemp";
        else if (kind == SPE_KIND_DIO && (strcasecmp(key, "perim") == 0 || strcasecmp(key, "peri") == 0))
            okey = "pj";

        sb_putc(&line, ' ');
        sb_puts(&line, okey);
        sb_putc(&line, '=');
        if ((is_prim_res || is_prim_cap || is_prim_ind) && spe_expr_has_vi_ref(val.buf))
            spe_format_beh_expr(&line, val.buf, TRUE);
        else
            spe_format_expr(&line, val.buf, FALSE);
    }

    sl_emit(out, line.buf);
    sb_free(&val);
    sb_free(&line);
    sb_free(&nodes);
}

/* =========================================================================
 * Helper: emit `include "file" [section=sec]` -> `.lib` / `.include`
 * ========================================================================= */
static void spe_emit_include(const char *s, SpeLines *out)
{
    const char *q = s + 7; /* skip "include" */
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
    sl_emit(out, line.buf);
    sb_free(&line);
}

/* =========================================================================
 * Recursive block emitter (handles if/elseif/else + conditional .param merge)
 * ========================================================================= */

static unsigned long spe_prev_uid = 0;

typedef struct {
    const char *cond;   /* NULL for `else` branch */
    size_t      b_start;
    size_t      b_end;
} SpeIfBranch;

static size_t spe_collect_if_branches(const SpeStmtList *stmts, size_t start, size_t limit,
                                      SpeIfBranch *branches, int max_branches, int *n_out)
{
    int nb = 0;
    int depth = 0;
    branches[0].cond = stmts->stmts[start] + 11; /* after "__SPE_IF__ " */
    branches[0].b_start = start + 1;
    nb = 1;

    size_t i = start + 1;
    for (; i < limit; i++) {
        if (stmts->is_spice[i]) continue;
        const char *s = stmts->stmts[i];
        if (strncmp(s, "__SPE_IF__ ", 11) == 0) {
            depth++;
        } else if (strcmp(s, "__SPE_ENDIF__") == 0) {
            if (depth == 0) {
                branches[nb - 1].b_end = i;
                *n_out = nb;
                return i;
            }
            depth--;
        } else if (depth == 0 && strncmp(s, "__SPE_ELSEIF__ ", 15) == 0) {
            branches[nb - 1].b_end = i;
            if (nb < max_branches) {
                branches[nb].cond = s + 15;
                branches[nb].b_start = i + 1;
                nb++;
            }
        } else if (depth == 0 && strcmp(s, "__SPE_ELSE__") == 0) {
            branches[nb - 1].b_end = i;
            if (nb < max_branches) {
                branches[nb].cond = NULL;
                branches[nb].b_start = i + 1;
                nb++;
            }
        }
    }
    branches[nb - 1].b_end = limit;
    *n_out = nb;
    return limit > start ? limit - 1 : start;
}

static bool spe_rename_prev_param(SpeLines *out, size_t scope_start, const char *key,
                                  char *alias_out, size_t alias_cap)
{
    size_t klen = strlen(key);
    for (size_t idx = out->count; idx > scope_start; idx--) {
        const char *ln = out->lines[idx - 1];
        if (strncasecmp(ln, ".param ", 7) == 0 &&
            strncasecmp(ln + 7, key, klen) == 0 &&
            ln[7 + klen] == '=') {
            snprintf(alias_out, alias_cap, "__spe_p_%lu", ++spe_prev_uid);
            SpeBuf rep; sb_init(&rep);
            sb_puts(&rep, ".param ");
            sb_puts(&rep, alias_out);
            sb_puts(&rep, ln + 7 + klen);
            tfree(out->lines[idx - 1]);
            out->lines[idx - 1] = copy(rep.buf);
            sb_free(&rep);
            return TRUE;
        }
    }
    if (scope_start > 0 && strncasecmp(out->lines[scope_start - 1], ".subckt ", 8) == 0) {
        const char *hdr = out->lines[scope_start - 1] + 8;
        for (const char *p = hdr; *p; p++) {
            if (isspace((unsigned char)*p)) {
                const char *q = p + 1;
                if (strncasecmp(q, key, klen) == 0 && q[klen] == '=') {
                    const char *v = q + klen + 1;
                    const char *ve = v;
                    while (*ve && !isspace((unsigned char)*ve)) ve++;
                    size_t vlen = (size_t)(ve - v);
                    if (vlen > 0 && vlen + 1 < alias_cap) {
                        memcpy(alias_out, v, vlen);
                        alias_out[vlen] = '\0';
                        return TRUE;
                    }
                }
            }
        }
    }
    return FALSE;
}

static void spe_emit_range(const SpeStmtList *stmts, size_t start, size_t end,
                           const char *dir_name, SpeSymTab *syms,
                           const char *cur_subckt_name, bool in_subckt,
                           bool is_top_level, size_t scope_out_start,
                           bool skip_params, SpeLines *out);

static void spe_emit_if_chain(const SpeStmtList *stmts, size_t start, size_t limit,
                              const char *dir_name, SpeSymTab *syms,
                              const char *cur_subckt_name, bool in_subckt,
                              bool is_top_level, size_t scope_out_start,
                              SpeLines *out, size_t *end_idx_out)
{
    SpeIfBranch branches[32];
    int nb = 0;
    size_t end_i = spe_collect_if_branches(stmts, start, limit, branches, 32, &nb);
    *end_idx_out = end_i;

    typedef struct {
        char *key;
        char *vals[32];
    } CondParam;
    CondParam *cparams = NULL;
    size_t n_cp = 0, cap_cp = 0;
    bool has_non_param = FALSE;

    for (int b = 0; b < nb; b++) {
        int nested = 0;
        for (size_t j = branches[b].b_start; j < branches[b].b_end; j++) {
            if (stmts->is_spice[j]) { has_non_param = TRUE; continue; }
            const char *s = stmts->stmts[j];
            while (*s && isspace((unsigned char)*s)) s++;
            if (!*s) continue;
            if (strncmp(s, "__SPE_IF__ ", 11) == 0) { nested++; has_non_param = TRUE; continue; }
            if (strcmp(s, "__SPE_ENDIF__") == 0) { if (nested > 0) nested--; continue; }
            if (nested > 0) continue;

            if (strncasecmp(s, "parameters", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]))) {
                const char *q = s + 10;
                char key[128];
                SpeBuf val; sb_init(&val);
                bool hv = FALSE;
                while (*q) {
                    if (!spe_next_kv(&q, key, sizeof(key), &val, &hv, FALSE) || !hv)
                        continue;
                    SpeBuf fmt; sb_init(&fmt);
                    spe_format_expr(&fmt, val.buf, FALSE);
                    const char *fe = fmt.buf;
                    size_t flen = fmt.len;
                    if (flen >= 2 && fe[0] == '{' && fe[flen - 1] == '}') {
                        fe++; flen -= 2;
                    }
                    size_t ci = 0;
                    for (; ci < n_cp; ci++) {
                        if (strcasecmp(cparams[ci].key, key) == 0) break;
                    }
                    if (ci == n_cp) {
                        if (n_cp >= cap_cp) {
                            cap_cp = cap_cp ? cap_cp * 2 : 32;
                            cparams = TREALLOC(CondParam, cparams, cap_cp);
                        }
                        cparams[ci].key = copy(key);
                        memset(cparams[ci].vals, 0, sizeof(cparams[ci].vals));
                        n_cp++;
                    }
                    tfree(cparams[ci].vals[b]);
                    cparams[ci].vals[b] = TMALLOC(char, flen + 1);
                    memcpy(cparams[ci].vals[b], fe, flen);
                    cparams[ci].vals[b][flen] = '\0';
                    sb_free(&fmt);
                }
                sb_free(&val);
            } else if (!spe_is_ignored_directive_stmt(s)) {
                has_non_param = TRUE;
            }
        }
    }

    for (size_t ci = 0; ci < n_cp; ci++) {
        bool has_else_val = (branches[nb - 1].cond == NULL && cparams[ci].vals[nb - 1] != NULL);
        char prev_alias[64] = "0";
        if (!has_else_val) {
            spe_rename_prev_param(out, scope_out_start, cparams[ci].key,
                                  prev_alias, sizeof(prev_alias));
        }

        SpeBuf acc; sb_init(&acc);
        if (has_else_val)
            sb_puts(&acc, cparams[ci].vals[nb - 1]);
        else
            sb_puts(&acc, prev_alias);

        for (int b = nb - 1; b >= 0; b--) {
            if (!branches[b].cond) continue;
            if (!cparams[ci].vals[b]) continue;
            SpeBuf cfmt; sb_init(&cfmt);
            spe_format_expr(&cfmt, branches[b].cond, FALSE);
            const char *ce = cfmt.buf;
            size_t clen = cfmt.len;
            if (clen >= 2 && ce[0] == '{' && ce[clen - 1] == '}') { ce++; clen -= 2; }

            SpeBuf next; sb_init(&next);
            sb_puts(&next, "((");
            sb_putn(&next, ce, clen);
            sb_puts(&next, ")?(");
            sb_puts(&next, cparams[ci].vals[b]);
            sb_puts(&next, "):(");
            sb_puts(&next, acc.buf);
            sb_puts(&next, "))");
            sb_free(&acc);
            acc = next;
            sb_free(&cfmt);
        }

        SpeBuf pline; sb_init(&pline);
        sb_puts(&pline, ".param ");
        sb_puts(&pline, cparams[ci].key);
        sb_puts(&pline, "={");
        sb_puts(&pline, acc.buf);
        sb_putc(&pline, '}');
        sl_emit(out, pline.buf);
        sb_free(&pline);
        sb_free(&acc);

        tfree(cparams[ci].key);
        for (int b = 0; b < nb; b++)
            tfree(cparams[ci].vals[b]);
    }
    tfree(cparams);

    if (has_non_param) {
        for (int b = 0; b < nb; b++) {
            SpeBuf hdr; sb_init(&hdr);
            if (b == 0) {
                sb_puts(&hdr, ".if (");
                SpeBuf cf; sb_init(&cf);
                spe_format_expr(&cf, branches[b].cond, FALSE);
                const char *ce = cf.buf; size_t clen = cf.len;
                if (clen >= 2 && ce[0] == '{' && ce[clen - 1] == '}') { ce++; clen -= 2; }
                sb_putn(&hdr, ce, clen);
                sb_putc(&hdr, ')');
                sb_free(&cf);
            } else if (branches[b].cond) {
                sb_puts(&hdr, ".elseif (");
                SpeBuf cf; sb_init(&cf);
                spe_format_expr(&cf, branches[b].cond, FALSE);
                const char *ce = cf.buf; size_t clen = cf.len;
                if (clen >= 2 && ce[0] == '{' && ce[clen - 1] == '}') { ce++; clen -= 2; }
                sb_putn(&hdr, ce, clen);
                sb_putc(&hdr, ')');
                sb_free(&cf);
            } else {
                sb_puts(&hdr, ".else");
            }
            sl_emit(out, hdr.buf);
            sb_free(&hdr);

            spe_emit_range(stmts, branches[b].b_start, branches[b].b_end,
                           dir_name, syms, cur_subckt_name, in_subckt,
                           is_top_level, scope_out_start, TRUE, out);
        }
        sl_emit(out, ".endif");
    }
}

static void spe_emit_range(const SpeStmtList *stmts, size_t start, size_t end,
                           const char *dir_name, SpeSymTab *syms,
                           const char *cur_subckt_name, bool in_subckt,
                           bool is_top_level, size_t scope_out_start,
                           bool skip_params, SpeLines *out)
{
    for (size_t i = start; i < end; i++) {
        if (stmts->is_spice[i]) {
            sl_emit(out, stmts->stmts[i]);
            continue;
        }
        const char *s = stmts->stmts[i];
        while (*s && isspace((unsigned char)*s)) s++;
        if (!*s)
            continue;

        if (strncmp(s, "__SPE_IF__ ", 11) == 0) {
            size_t end_i = i;
            spe_emit_if_chain(stmts, i, end, dir_name, syms, cur_subckt_name,
                              in_subckt, is_top_level, scope_out_start, out, &end_i);
            i = end_i;
            continue;
        }
        if (strcmp(s, "__SPE_ENDIF__") == 0 || strcmp(s, "__SPE_ELSE__") == 0 ||
            strncmp(s, "__SPE_ELSEIF__ ", 15) == 0)
            continue;

        if (spe_is_ignored_directive_stmt(s))
            continue;

        /* User-defined Spectre `real funcname(...) { return ... }` -> `.func` */
        if (strncasecmp(s, "real", 4) == 0 && isspace((unsigned char)s[4])) {
            spe_translate_real_func(s, out);
            continue;
        }

        /* section <name> -> .lib <name> */
        if (!in_subckt && strncasecmp(s, "section", 7) == 0 && isspace((unsigned char)s[7])) {
            const char *q = s + 7;
            while (*q && isspace((unsigned char)*q)) q++;
            SpeBuf line; sb_init(&line);
            sb_puts(&line, ".lib ");
            sb_puts(&line, q);
            sl_emit(out, line.buf);
            sb_free(&line);
            continue;
        }

        /* endsection [name] -> .endl [name] */
        if (!in_subckt && strncasecmp(s, "endsection", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]))) {
            const char *q = s + 10;
            while (*q && isspace((unsigned char)*q)) q++;
            SpeBuf line; sb_init(&line);
            sb_puts(&line, ".endl");
            if (*q) {
                sb_putc(&line, ' ');
                sb_puts(&line, q);
            }
            sl_emit(out, line.buf);
            sb_free(&line);
            continue;
        }

        /* include "file" [section=sec] -> .lib / .include */
        if (strncasecmp(s, "include", 7) == 0 && isspace((unsigned char)s[7])) {
            spe_emit_include(s, out);
            continue;
        }

        /* parameters ... */
        if (strncasecmp(s, "parameters", 10) == 0 && (!s[10] || isspace((unsigned char)s[10]))) {
            if (skip_params)
                continue;
            const char *q = s + 10;
            char key[128];
            SpeBuf val; sb_init(&val);
            bool hv = FALSE;
            while (*q) {
                if (!spe_next_kv(&q, key, sizeof(key), &val, &hv, in_subckt))
                    continue;
                if (in_subckt) {
                    if (strcasecmp(key, "m") == 0)
                        continue;
                    if (!hv || is_plain_number(val.buf) || is_simple_val(val.buf))
                        continue;
                } else {
                    if (!hv) continue;
                }
                SpeBuf line; sb_init(&line);
                sb_puts(&line, ".param ");
                sb_puts(&line, key);
                sb_putc(&line, '=');
                spe_format_expr(&line, val.buf, FALSE);
                sl_emit(out, line.buf);
                sb_free(&line);
            }
            sb_free(&val);
            continue;
        }

        /* model ... */
        if (strncasecmp(s, "model", 5) == 0 && isspace((unsigned char)s[5])) {
            spe_translate_model(s, dir_name, out);
            continue;
        }

        /* [inline] subckt name [(ports...)] ... ends [name] */
        if (!in_subckt &&
            ((strncasecmp(s, "inline", 6) == 0 && isspace((unsigned char)s[6])) ||
             (strncasecmp(s, "subckt", 6) == 0 && isspace((unsigned char)s[6])))) {
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

                size_t end_i = i + 1;
                sym_clear_local(syms);
                while (end_i < end) {
                    if (!stmts->is_spice[end_i]) {
                        const char *es = stmts->stmts[end_i];
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
                                sym_add_ex(syms, mn, kd, spe_cmc_osdi_type(mt), 1);
                        } else if (strncasecmp(es, "include", 7) == 0 && isspace((unsigned char)es[7])) {
                            const char *iq = es + 7;
                            while (*iq && isspace((unsigned char)*iq)) iq++;
                            char incf[1024];
                            size_t ik = 0;
                            if (*iq == '"' || *iq == '\'') {
                                char qt = *iq++;
                                while (*iq && *iq != qt && ik + 1 < sizeof(incf)) incf[ik++] = *iq++;
                            } else {
                                while (*iq && !isspace((unsigned char)*iq) && ik + 1 < sizeof(incf)) incf[ik++] = *iq++;
                            }
                            incf[ik] = '\0';
                            if (incf[0])
                                spe_prescan_models_in_file(incf, dir_name, 0);
                        }
                    }
                    end_i++;
                }

                SpeBuf sub_hdr; sb_init(&sub_hdr);
                sb_puts(&sub_hdr, ".subckt ");
                sb_puts(&sub_hdr, sname);
                if (ports.len > 0) {
                    sb_putc(&sub_hdr, ' ');
                    sb_puts(&sub_hdr, ports.buf);
                }

                int sub_if_depth = 0;
                for (size_t j = i + 1; j < end_i; j++) {
                    if (stmts->is_spice[j]) continue;
                    const char *ps = stmts->stmts[j];
                    while (*ps && isspace((unsigned char)*ps)) ps++;
                    if (strncmp(ps, "__SPE_IF__ ", 11) == 0) { sub_if_depth++; continue; }
                    if (strcmp(ps, "__SPE_ENDIF__") == 0) { if (sub_if_depth > 0) sub_if_depth--; continue; }
                    if (sub_if_depth > 0) continue;
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
                        if (strcasecmp(key, "m") == 0)
                            continue;
                        if (!hv) {
                            sb_putc(&sub_hdr, ' ');
                            sb_puts(&sub_hdr, key);
                            sb_puts(&sub_hdr, "=0");
                        } else if (is_plain_number(val.buf) || is_simple_val(val.buf)) {
                            sb_putc(&sub_hdr, ' ');
                            sb_puts(&sub_hdr, key);
                            sb_putc(&sub_hdr, '=');
                            sb_puts(&sub_hdr, val.buf);
                        }
                    }
                    sb_free(&val);
                }

                sl_emit(out, sub_hdr.buf);
                sb_free(&sub_hdr);
                sb_free(&ports);

                size_t sub_scope_start = out->count;
                spe_emit_range(stmts, i + 1, end_i, dir_name, syms, sname,
                               TRUE, FALSE, sub_scope_start, FALSE, out);

                SpeBuf end_l; sb_init(&end_l);
                sb_puts(&end_l, ".ends ");
                sb_puts(&end_l, sname);
                sl_emit(out, end_l.buf);
                sb_free(&end_l);

                sym_clear_local(syms);
                i = end_i;
                continue;
            }
        }

        /* global <nodes...> -> .global <non-zero nodes...> */
        if (!in_subckt && strncasecmp(s, "global", 6) == 0 && isspace((unsigned char)s[6])) {
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
                sl_emit(out, line.buf);
            sb_free(&line);
            continue;
        }

        /* Top-level Spectre analysis statements: `<name> dc|tran|ac ...` */
        if (!in_subckt && is_top_level && !strchr(s, '(')) {
            const char *q = s;
            while (*q && !isspace((unsigned char)*q)) q++;
            while (*q && isspace((unsigned char)*q)) q++;
            if (strncasecmp(q, "dc", 2) == 0 && (!q[2] || isspace((unsigned char)q[2]))) {
                sl_emit(out, ".op");
                continue;
            }
            if (strncasecmp(q, "tran", 4) == 0 && (!q[4] || isspace((unsigned char)q[4]))) {
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
                    sl_emit(out, line.buf);
                    sb_free(&line);
                }
                sb_free(&stop_v); sb_free(&step_v); sb_free(&val);
                continue;
            }
            if (strncasecmp(q, "ac", 2) == 0 && (!q[2] || isspace((unsigned char)q[2]))) {
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
                    sl_emit(out, line.buf);
                    sb_free(&line);
                }
                sb_free(&start_v); sb_free(&stop_v); sb_free(&dec_v); sb_free(&val);
                continue;
            }
        }

        /* Spectre device instance (parenthesized OR unparenthesized) */
        spe_translate_instance(s, syms, cur_subckt_name, out);
    }
}

/* =========================================================================
 * Public detection & translation entry points
 * ========================================================================= */

bool inp_spectre_should_translate(FILE *fp, const char *file_name)
{
    if (!fp)
        return FALSE;

    if (file_name) {
        size_t n = strlen(file_name);
        if (n >= 4 && strcasecmp(file_name + n - 4, ".scs") == 0)
            return TRUE;
    }

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

    newcompat.spe = TRUE;
    newcompat.isset = TRUE;

    SpeStmtList stmts;
    st_init(&stmts);
    spe_read_statements(fp, &stmts);

    /* Pass 1: collect global symbol table (models, subckts, ahdl_include VA modules,
     *         and prescan referenced include files for cross-file model kinds) */
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
        if (strncasecmp(s, "include", 7) == 0 && isspace((unsigned char)s[7])) {
            const char *q = s + 7;
            while (*q && isspace((unsigned char)*q)) q++;
            char incf[1024];
            size_t k = 0;
            if (*q == '"' || *q == '\'') {
                char qt = *q++;
                while (*q && *q != qt && k + 1 < sizeof(incf)) incf[k++] = *q++;
            } else {
                while (*q && !isspace((unsigned char)*q) && k + 1 < sizeof(incf)) incf[k++] = *q++;
            }
            incf[k] = '\0';
            if (incf[0])
                spe_prescan_models_in_file(incf, dir_name, 0);
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
            if (kd != SPE_KIND_UNKNOWN) {
                const char *cmc = spe_cmc_osdi_type(mtype);
                sym_add_ex(&syms, mname, kd, cmc, 0);
                spe_register_global_model(mname, kd, cmc);
            }
        }
    }

    /* Pass 2: emit translated SPICE lines */
    SpeLines out;
    sl_init(&out);

    if (is_top_level) {
        SpeBuf title; sb_init(&title);
        sb_puts(&title, "* Spectre deck: ");
        sb_puts(&title, file_name ? file_name : "(stdin)");
        sl_emit(&out, title.buf);
        sb_free(&title);
    }

    spe_emit_range(&stmts, 0, stmts.count, dir_name, &syms, NULL,
                   FALSE, is_top_level, out.count, FALSE, &out);

    if (is_top_level)
        sl_emit(&out, ".end");

    sym_free(&syms);
    st_free(&stmts);

    *out_nlines = out.count;
    return out.lines;
}
