/*
 * nglib_cache.h — Explicit PDK / Library Precompiler & Binary Blob Loader
 *
 * Commands / Directives:
 *   CLI:      ngspice --precompile <out.nglib> <in_pdk.spice>
 *   Control:  precompile <out.nglib> <in_pdk.spice>
 *   Netlist:  .uselib "<path/to/pdk.nglib>"
 *
 * Precompiles a PDK library deck (all .lib/.include expansions, line joins,
 * compatibility transforms, top-level .model parameter evaluation, constant
 * folding of zero-switch mismatch expressions in .subckt bodies, and dead
 * local .param elimination) into a compact binary blob (magic "NGLIB001")
 * for rapid reuse across changing top-level netlists.
 */

#ifndef NGSPICE_NGLIB_CACHE_H
#define NGSPICE_NGLIB_CACHE_H

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "subckt.h"

static double
nglib_now_ms(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec * 1e-6;
}

static void
nglib_free_cards(struct card *c)
{
    while (c) {
        struct card *next = c->nextcard;
        if (c->actualLine)
            nglib_free_cards(c->actualLine);
        tfree(c->line);
        tfree(c->error);
        tfree(c);
        c = next;
    }
}

#define NGLIB_MAGIC      "NGLIB001"
#define NGLIB_MAGIC_LEN  8
#define NGLIB_VERSION    1u

typedef struct {
    char     magic[NGLIB_MAGIC_LEN];
    uint32_t version;
    uint32_t compat_mask;
    uint32_t num_cards;
    uint64_t total_str_bytes;
} NglibHeader;

static uint32_t
nglib_current_compat_mask(void)
{
    uint32_t m = 0;
    if (newcompat.hs)  m |= (1u << 0);
    if (newcompat.spe) m |= (1u << 1);
    if (newcompat.ps)  m |= (1u << 2);
    if (newcompat.lt)  m |= (1u << 3);
    if (newcompat.s3)  m |= (1u << 4);
    if (newcompat.a)   m |= (1u << 5);
    if (newcompat.ll)  m |= (1u << 6);
    if (newcompat.ki)  m |= (1u << 7);
    if (newcompat.xs)  m |= (1u << 8);
    return m;
}

static int
nglib_is_ident_char(int c)
{
    return isalnum((unsigned char)c) || c == '_';
}

static int
nglib_contains_ident(const char *haystack, const char *ident)
{
    size_t n = strlen(ident);
    if (n == 0 || !haystack)
        return 0;
    const char *p = haystack;
    while ((p = strstr(p, ident)) != NULL) {
        int left_ok  = (p == haystack) || !nglib_is_ident_char((unsigned char)p[-1]);
        int right_ok = !nglib_is_ident_char((unsigned char)p[n]);
        if (left_ok && right_ok)
            return 1;
        p += n;
    }
    return 0;
}

typedef struct {
    char *name;
    char *rhs;
    int   live;
} NglibParamAssign;

static int
nglib_is_literal_zero(const char *s)
{
    while (*s && isspace((unsigned char)*s)) s++;
    if (*s == '{' || *s == '\'' || *s == '(') {
        char close = (*s == '{') ? '}' : ((*s == '(') ? ')' : '\'');
        s++;
        while (*s && isspace((unsigned char)*s)) s++;
        char *end = NULL;
        double v = strtod(s, &end);
        if (end == s) return 0;
        while (*end && isspace((unsigned char)*end)) end++;
        if (*end != close) return 0;
        end++;
        while (*end && isspace((unsigned char)*end)) end++;
        return (*end == '\0' && v == 0.0);
    }
    char *end = NULL;
    double v = strtod(s, &end);
    if (end == s) return 0;
    while (*end && isspace((unsigned char)*end)) end++;
    return (*end == '\0' && v == 0.0);
}

/* Split a ".param a=1 b='x+y' c={z}" line into individual (name, rhs) pairs */
static void
nglib_parse_param_line(const char *line, NglibParamAssign **arr, int *cnt, int *cap)
{
    const char *p = line;
    while (*p && isspace((unsigned char)*p)) p++;
    if (!ciprefix(".param", p))
        return;
    p += 6;
    /* Handle optional .parameters */
    while (*p && !isspace((unsigned char)*p)) p++;

    while (*p) {
        while (*p && (isspace((unsigned char)*p) || *p == ',')) p++;
        if (!*p) break;

        /* Read parameter name */
        const char *ns = p;
        while (*p && nglib_is_ident_char((unsigned char)*p)) p++;
        size_t nlen = (size_t)(p - ns);
        if (nlen == 0) {
            /* Skip unexpected token */
            while (*p && !isspace((unsigned char)*p)) p++;
            continue;
        }
        while (*p && isspace((unsigned char)*p)) p++;
        if (*p != '=') {
            /* Not an assignment; skip */
            continue;
        }
        p++; /* skip '=' */
        while (*p && isspace((unsigned char)*p)) p++;
        if (!*p) break;

        /* Read RHS respecting quotes, braces, parentheses */
        const char *rs = p;
        int depth_p = 0, depth_b = 0;
        char in_q = 0;
        while (*p) {
            char c = *p;
            if (in_q) {
                if (c == in_q) in_q = 0;
                p++;
                continue;
            }
            if (c == '\'' || c == '"') {
                in_q = c;
                p++;
                continue;
            }
            if (c == '{') { depth_b++; p++; continue; }
            if (c == '}') { if (depth_b > 0) depth_b--; p++; continue; }
            if (c == '(') { depth_p++; p++; continue; }
            if (c == ')') { if (depth_p > 0) depth_p--; p++; continue; }
            if (depth_b == 0 && depth_p == 0 && (isspace((unsigned char)c) || c == ',')) {
                /* Peek ahead: if next non-space token starts an ident followed by '=', stop */
                const char *q = p;
                while (*q && (isspace((unsigned char)*q) || *q == ',')) q++;
                if (!*q) break;
                const char *q2 = q;
                while (*q2 && nglib_is_ident_char((unsigned char)*q2)) q2++;
                if (q2 > q) {
                    const char *q3 = q2;
                    while (*q3 && isspace((unsigned char)*q3)) q3++;
                    if (*q3 == '=')
                        break;
                }
                /* Also stop if RHS already ended with '}' or '\'' */
                if (p > rs && (p[-1] == '}' || p[-1] == '\'' || p[-1] == '"'))
                    break;
            }
            p++;
        }
        const char *re = p;
        while (re > rs && isspace((unsigned char)re[-1])) re--;

        if (*cnt == *cap) {
            int ncap = *cap ? (*cap * 2) : 32;
            *arr = TREALLOC(NglibParamAssign, *arr, ncap);
            *cap = ncap;
        }
        NglibParamAssign *a = &(*arr)[(*cnt)++];
        a->name = copy_substring(ns, ns + nlen);
        a->rhs  = copy_substring(rs, re);
        a->live = 0;
    }
}

/* Check if trimmed expression [s, e) is a multiplicative product containing an unshadowed zero param */
static int
nglib_expr_is_zero_prod(const char *s, const char *e,
                        char **zeros, int nzeros,
                        char **shadow, int nshadow)
{
    while (s < e && isspace((unsigned char)*s)) s++;
    while (e > s && isspace((unsigned char)e[-1])) e--;
    if (s >= e) return 0;

    /* Must not have top-level '+' or '-' (except unary leading sign) */
    int depth = 0;
    int has_mul = 0;
    for (const char *p = s; p < e; p++) {
        if (*p == '(') depth++;
        else if (*p == ')') { if (depth > 0) depth--; }
        else if (depth == 0) {
            if ((*p == '+' || *p == '-') && p > s)
                return 0;
            if (*p == '*')
                has_mul = 1;
        }
    }
    if (!has_mul)
        return 0;

    /* Split at top-level '*' and check if any bare factor is a known global zero (not shadowed) */
    const char *fs = s;
    depth = 0;
    for (const char *p = s; p <= e; p++) {
        if (p < e && *p == '(') depth++;
        else if (p < e && *p == ')') { if (depth > 0) depth--; }
        else if (p == e || (depth == 0 && *p == '*')) {
            const char *a = fs, *b = p;
            while (a < b && isspace((unsigned char)*a)) a++;
            while (b > a && isspace((unsigned char)b[-1])) b--;
            size_t flen = (size_t)(b - a);
            if (flen > 0) {
                int is_z = 0;
                for (int zi = 0; zi < nzeros; zi++) {
                    if (strlen(zeros[zi]) == flen && memcmp(zeros[zi], a, flen) == 0) {
                        is_z = 1;
                        break;
                    }
                }
                if (is_z) {
                    int is_sh = 0;
                    for (int si = 0; si < nshadow; si++) {
                        if (strlen(shadow[si]) == flen && memcmp(shadow[si], a, flen) == 0) {
                            is_sh = 1;
                            break;
                        }
                    }
                    if (!is_sh)
                        return 1;
                }
            }
            fs = p + 1;
        }
    }
    return 0;
}

/* Fold {expr * Z} -> 0 and {1 +- expr * Z} -> 1 inside a single card line */
static char *
nglib_fold_zero_switches_line(const char *line,
                              char **zeros, int nzeros,
                              char **shadow, int nshadow,
                              int *folded_count)
{
    if (nzeros == 0 || !line)
        return copy(line);

    size_t len = strlen(line);
    char *out = TMALLOC(char, len + 16);
    size_t oi = 0;
    const char *p = line;

    while (*p) {
        if (*p == '{' || *p == '\'') {
            char open = *p;
            char close = (open == '{') ? '}' : '\'';
            const char *es = p + 1;
            const char *ee = es;
            int depth = 0;
            while (*ee) {
                if (open == '{') {
                    if (*ee == '{') depth++;
                    else if (*ee == '}') {
                        if (depth == 0) break;
                        depth--;
                    }
                } else {
                    if (*ee == '\'') break;
                }
                ee++;
            }
            if (*ee == close) {
                const char *ts = es, *te = ee;
                while (ts < te && isspace((unsigned char)*ts)) ts++;
                while (te > ts && isspace((unsigned char)te[-1])) te--;

                if (nglib_expr_is_zero_prod(ts, te, zeros, nzeros, shadow, nshadow)) {
                    out[oi++] = '0';
                    if (folded_count) (*folded_count)++;
                    p = ee + 1;
                    continue;
                }
                /* Check "1 - <zero_prod>" or "1 + <zero_prod>" */
                if (te > ts + 2 && ts[0] == '1') {
                    const char *q = ts + 1;
                    while (q < te && isspace((unsigned char)*q)) q++;
                    if (q < te && (*q == '-' || *q == '+')) {
                        q++;
                        if (nglib_expr_is_zero_prod(q, te, zeros, nzeros, shadow, nshadow)) {
                            out[oi++] = '1';
                            if (folded_count) (*folded_count)++;
                            p = ee + 1;
                            continue;
                        }
                    }
                }
            }
        }
        out[oi++] = *p++;
    }
    out[oi] = '\0';
    return out;
}

/* Simplify one .subckt ... .ends block: fold zero switches + eliminate dead local .param */
static void
nglib_optimize_subckt_block(struct card *sub_head, struct card *sub_tail,
                            char **zeros, int nzeros,
                            int *folded_exprs, int *dead_params)
{
    /* Skip nested subckts conservatively */
    int depth = 0;
    for (struct card *c = sub_head; c; c = c->nextcard) {
        if (ciprefix(".subckt", c->line)) depth++;
        if (depth > 1) return;
        if (c == sub_tail) break;
    }

    /* Collect shadowed names from .subckt header (params & ports) and local .param lines */
    char **shadow = NULL;
    int nshadow = 0, cap_shadow = 0;

    /* Header tokens after ".subckt <name>" */
    {
        const char *p = sub_head->line;
        while (*p && isspace((unsigned char)*p)) p++;
        while (*p && !isspace((unsigned char)*p)) p++; /* .subckt */
        while (*p && isspace((unsigned char)*p)) p++;
        while (*p && !isspace((unsigned char)*p)) p++; /* name */
        while (*p) {
            while (*p && isspace((unsigned char)*p)) p++;
            if (!*p) break;
            const char *ts = p;
            while (*p && nglib_is_ident_char((unsigned char)*p)) p++;
            if (p > ts) {
                if (nshadow == cap_shadow) {
                    cap_shadow = cap_shadow ? cap_shadow * 2 : 32;
                    shadow = TREALLOC(char *, shadow, cap_shadow);
                }
                shadow[nshadow++] = copy_substring(ts, p);
            }
            /* Skip '=' and default value if present */
            while (*p && isspace((unsigned char)*p)) p++;
            if (*p == '=') {
                p++;
                while (*p && isspace((unsigned char)*p)) p++;
                int dp = 0, db = 0;
                char iq = 0;
                while (*p) {
                    if (iq) { if (*p == iq) iq = 0; p++; continue; }
                    if (*p == '\'' || *p == '"') { iq = *p++; continue; }
                    if (*p == '{') { db++; p++; continue; }
                    if (*p == '}') { if (db > 0) db--; p++; continue; }
                    if (*p == '(') { dp++; p++; continue; }
                    if (*p == ')') { if (dp > 0) dp--; p++; continue; }
                    if (dp == 0 && db == 0 && isspace((unsigned char)*p)) break;
                    p++;
                }
            } else if (*p && !isspace((unsigned char)*p)) {
                while (*p && !isspace((unsigned char)*p)) p++;
            }
        }
    }

    /* Parse all local .param assignments inside this .subckt */
    NglibParamAssign *locals = NULL;
    int nlocals = 0, cap_locals = 0;
    for (struct card *c = sub_head->nextcard; c && c != sub_tail; c = c->nextcard) {
        if (ciprefix(".param", c->line))
            nglib_parse_param_line(c->line, &locals, &nlocals, &cap_locals);
    }
    for (int i = 0; i < nlocals; i++) {
        if (nshadow == cap_shadow) {
            cap_shadow = cap_shadow ? cap_shadow * 2 : 32;
            shadow = TREALLOC(char *, shadow, cap_shadow);
        }
        shadow[nshadow++] = copy(locals[i].name);
    }

    /* Fold zero-switch expressions on all body lines */
    for (struct card *c = sub_head->nextcard; c && c != sub_tail; c = c->nextcard) {
        if (!ciprefix(".param", c->line)) {
            char *nl = nglib_fold_zero_switches_line(c->line, zeros, nzeros, shadow, nshadow, folded_exprs);
            tfree(c->line);
            c->line = nl;
        }
    }

    /* Dead local .param elimination */
    if (nlocals > 0) {
        /* Also fold zero-switches inside local .param RHS */
        for (int i = 0; i < nlocals; i++) {
            char *nr = nglib_fold_zero_switches_line(locals[i].rhs, zeros, nzeros, shadow, nshadow, folded_exprs);
            tfree(locals[i].rhs);
            locals[i].rhs = nr;
        }

        /* Seed liveness from non-.param body lines */
        for (struct card *c = sub_head->nextcard; c && c != sub_tail; c = c->nextcard) {
            if (ciprefix(".param", c->line))
                continue;
            for (int i = 0; i < nlocals; i++) {
                if (!locals[i].live && nglib_contains_ident(c->line, locals[i].name))
                    locals[i].live = 1;
            }
        }
        /* Fixed-point backward liveness across local .param dependencies */
        int changed = 1;
        while (changed) {
            changed = 0;
            for (int j = 0; j < nlocals; j++) {
                if (!locals[j].live) continue;
                for (int i = 0; i < nlocals; i++) {
                    if (!locals[i].live && nglib_contains_ident(locals[j].rhs, locals[i].name)) {
                        locals[i].live = 1;
                        changed = 1;
                    }
                }
            }
        }

        int n_dead = 0, n_live = 0;
        for (int i = 0; i < nlocals; i++) {
            if (locals[i].live) n_live++;
            else n_dead++;
        }
        if (n_dead > 0) {
            if (dead_params) *dead_params += n_dead;
            int emitted_live = 0;
            for (struct card *c = sub_head->nextcard; c && c != sub_tail; c = c->nextcard) {
                if (!ciprefix(".param", c->line))
                    continue;
                if (!emitted_live && n_live > 0) {
                    /* Rebuild a single compact .param line with only live locals */
                    size_t tot = 8;
                    for (int i = 0; i < nlocals; i++)
                        if (locals[i].live)
                            tot += strlen(locals[i].name) + strlen(locals[i].rhs) + 3;
                    char *buf = TMALLOC(char, tot + 1);
                    strcpy(buf, ".param");
                    for (int i = 0; i < nlocals; i++) {
                        if (locals[i].live) {
                            strcat(buf, " ");
                            strcat(buf, locals[i].name);
                            strcat(buf, "=");
                            strcat(buf, locals[i].rhs);
                        }
                    }
                    tfree(c->line);
                    c->line = buf;
                    emitted_live = 1;
                } else {
                    /* Mark dead .param card as comment to be stripped before write */
                    tfree(c->line);
                    c->line = copy("*");
                }
            }
        }
    }

    for (int i = 0; i < nlocals; i++) {
        tfree(locals[i].name);
        tfree(locals[i].rhs);
    }
    FREE(locals);
    for (int i = 0; i < nshadow; i++)
        tfree(shadow[i]);
    FREE(shadow);
}

static struct card *
nglib_append_card(struct card **head, struct card **tail, const char *line, int lnum)
{
    struct card *c = TMALLOC(struct card, 1);
    c->linenum = lnum;
    c->linenum_orig = lnum;
    c->linesource = NULL;
    c->line = copy(line);
    c->error = NULL;
    c->nextcard = NULL;
    c->actualLine = NULL;
    c->compmod = 0;
    if (*tail)
        (*tail)->nextcard = c;
    else
        *head = c;
    *tail = c;
    return c;
}

int
nglib_precompile(const char *in_file, const char *out_file)
{
    if (!in_file || !out_file) {
        fprintf(cp_err, "Error: precompile requires <out.nglib> <in_pdk.spice>\n");
        return 1;
    }

    double t0 = nglib_now_ms();
    FILE *fp = fopen(in_file, "r");
    if (!fp) {
        fprintf(cp_err, "Error: precompile cannot open input file '%s'\n", in_file);
        return 1;
    }

    char *dir_name = ngdirname(in_file);
    bool expr_w_temper = FALSE;
    nglib_precompiling = 1;
    struct card *deck = inp_readall(fp, dir_name, in_file, FALSE, FALSE, &expr_w_temper);
    nglib_precompiling = 0;
    fclose(fp);
    tfree(dir_name);

    if (!deck) {
        fprintf(cp_err, "Error: precompile got empty deck from '%s'\n", in_file);
        return 1;
    }

    /* Separate top-level cards (outside .subckt) and .subckt ... .ends blocks */
    struct card *top_head = NULL, *top_tail = NULL;
    struct card *sub_head = NULL, *sub_tail = NULL;
    int sub_depth = 0;
    int lnum = 1;

    /* Skip title card (first card) */
    for (struct card *c = deck->nextcard; c; c = c->nextcard) {
        if (!c->line) continue;
        char *s = skip_ws(c->line);
        if (!*s || *s == '*') continue;
        if (ciprefix(".end", s) && !ciprefix(".ends", s) && !ciprefix(".endl", s) &&
            (s[4] == '\0' || isspace((unsigned char)s[4])))
            continue;
        if (ciprefix(".title", s))
            continue;

        if (ciprefix(".subckt", s)) {
            sub_depth++;
            nglib_append_card(&sub_head, &sub_tail, s, lnum++);
        } else if (ciprefix(".ends", s)) {
            if (sub_depth > 0) sub_depth--;
            nglib_append_card(&sub_head, &sub_tail, s, lnum++);
        } else if (sub_depth > 0) {
            nglib_append_card(&sub_head, &sub_tail, s, lnum++);
        } else {
            nglib_append_card(&top_head, &top_tail, s, lnum++);
        }
    }
    nglib_free_cards(deck);

    /* 1. Identify literal zero global .param names from top_head */
    NglibParamAssign *gparams = NULL;
    int ngparams = 0, cap_gparams = 0;
    for (struct card *c = top_head; c; c = c->nextcard) {
        if (ciprefix(".param", c->line))
            nglib_parse_param_line(c->line, &gparams, &ngparams, &cap_gparams);
    }

    char **zeros = NULL;
    int nzeros = 0, cap_zeros = 0;
    for (int i = 0; i < ngparams; i++) {
        /* Last assignment wins */
        int is_last = 1;
        for (int j = i + 1; j < ngparams; j++) {
            if (strcmp(gparams[i].name, gparams[j].name) == 0) {
                is_last = 0;
                break;
            }
        }
        if (is_last && nglib_is_literal_zero(gparams[i].rhs)) {
            if (nzeros == cap_zeros) {
                cap_zeros = cap_zeros ? cap_zeros * 2 : 16;
                zeros = TREALLOC(char *, zeros, cap_zeros);
            }
            zeros[nzeros++] = copy(gparams[i].name);
        }
    }
    for (int i = 0; i < ngparams; i++) {
        tfree(gparams[i].name);
        tfree(gparams[i].rhs);
    }
    FREE(gparams);

    /* 2. Evaluate top-level .model cards via inp_subcktexpand on a synthetic deck.
     *    Run inp_fix_for_numparam on synth_head ONLY so single-quoted HSPICE
     *    expressions are converted to {braces} for numparam without touching sub_head. */
    struct card *synth_head = NULL, *synth_tail = NULL;
    nglib_append_card(&synth_head, &synth_tail, "* nglib_precompile_eval", 0);
    int sln = 1;
    for (struct card *c = top_head; c; c = c->nextcard) {
        struct card *sc = nglib_append_card(&synth_head, &synth_tail, c->line, sln++);
        sc->linesource = c->linesource;
    }
    if (synth_head->nextcard) {
        static char *statfcn[] = { "agauss", "gauss", "aunif", "unif", "limit" };
        struct names *tmp_swp = new_names();
        inp_fix_for_numparam(tmp_swp, synth_head->nextcard);
        inp_remove_excess_ws(synth_head->nextcard);
        sln = inp_split_multi_param_lines(synth_head->nextcard, sln);
        inp_fix_macro_param_func_paren_io(synth_head->nextcard);
        for (int ii = 0; ii < 5; ii++)
            inp_fix_agauss_in_param(synth_head->nextcard, statfcn[ii]);
        inp_fix_temper_in_param(synth_head->nextcard);
        inp_expand_macros_in_deck(NULL, synth_head->nextcard);
        inp_fix_param_values(synth_head->nextcard);
        inp_reorder_params(tmp_swp, synth_head);
        delete_names(tmp_swp);
    }
    /* Count cards for dynmaxline (inp_split_multi_param_lines adds cards) */
    {
        int ncards_synth = 0;
        for (struct card *c = synth_head; c; c = c->nextcard)
            ncards_synth++;
        dynmaxline = ncards_synth + 64;
    }
    struct card *eval_top = inp_subcktexpand(synth_head->nextcard);
    synth_head->nextcard = eval_top;

    /* 3. Optimize every .subckt ... .ends block (zero-switch fold + dead local .param elimination) */
    int folded_exprs = 0, dead_params = 0, n_subckts = 0, n_models = 0;
    {
        struct card *c = sub_head;
        while (c) {
            if (ciprefix(".subckt", c->line)) {
                struct card *blk_head = c;
                int d = 1;
                struct card *blk_tail = c;
                while (blk_tail->nextcard && d > 0) {
                    blk_tail = blk_tail->nextcard;
                    if (ciprefix(".subckt", blk_tail->line)) d++;
                    else if (ciprefix(".ends", blk_tail->line)) d--;
                }
                nglib_optimize_subckt_block(blk_head, blk_tail, zeros, nzeros,
                                            &folded_exprs, &dead_params);
                n_subckts++;
                c = blk_tail->nextcard;
            } else {
                c = c->nextcard;
            }
        }
    }

    /* 4. Assemble final compact card list:
     *    - live top-level .param cards (cross-card liveness resolved globally)
     *    - top-level .func/.global/.options cards from top_head
     *    - fully evaluated .model cards from eval_top (zero expressions left!)
     *    - optimized .subckt ... .ends blocks from sub_head
     */
    struct card *out_head = NULL, *out_tail = NULL;
    uint32_t num_cards = 0;
    uint64_t total_str_bytes = 0;

    {
        NglibParamAssign *pa = NULL;
        int npa = 0, cpa = 0;
        for (struct card *c = top_head; c; c = c->nextcard) {
            if (c->line && c->line[0] != '*' && ciprefix(".param", c->line))
                nglib_parse_param_line(c->line, &pa, &npa, &cpa);
        }
        for (int i = 0; i < npa; i++) {
            if (nglib_is_literal_zero(pa[i].rhs)) {
                pa[i].live = 1;
                continue;
            }
            for (struct card *sc = sub_head; sc; sc = sc->nextcard) {
                if (sc->line && sc->line[0] != '*' && nglib_contains_ident(sc->line, pa[i].name)) {
                    pa[i].live = 1;
                    break;
                }
            }
        }
        int changed = 1;
        while (changed) {
            changed = 0;
            for (int j = 0; j < npa; j++) {
                if (!pa[j].live) continue;
                for (int i = 0; i < npa; i++) {
                    if (!pa[i].live && nglib_contains_ident(pa[j].rhs, pa[i].name)) {
                        pa[i].live = 1;
                        changed = 1;
                    }
                }
            }
        }
        for (int i = 0; i < npa; i++) {
            if (pa[i].live) {
                char *buf = tprintf(".param %s=%s", pa[i].name, pa[i].rhs);
                nglib_append_card(&out_head, &out_tail, buf, (int)num_cards + 1);
                total_str_bytes += strlen(buf) + 1;
                num_cards++;
                tfree(buf);
            }
            tfree(pa[i].name);
            tfree(pa[i].rhs);
        }
        FREE(pa);
    }

    for (struct card *c = top_head; c; c = c->nextcard) {
        if (!c->line || c->line[0] == '*') continue;
        if (ciprefix(".model", c->line) || ciprefix(".param", c->line)) continue;
        nglib_append_card(&out_head, &out_tail, c->line, (int)num_cards + 1);
        total_str_bytes += strlen(c->line) + 1;
        num_cards++;
    }

    for (struct card *c = eval_top; c; c = c->nextcard) {
        if (!c->line) continue;
        char *s = skip_ws(c->line);
        if (!*s || *s == '*') continue;
        if (ciprefix(".model", s)) {
            nglib_append_card(&out_head, &out_tail, s, (int)num_cards + 1);
            total_str_bytes += strlen(s) + 1;
            num_cards++;
            n_models++;
        }
    }

    for (struct card *c = sub_head; c; c = c->nextcard) {
        if (!c->line) continue;
        char *s = skip_ws(c->line);
        if (!*s || *s == '*') continue;
        nglib_append_card(&out_head, &out_tail, s, (int)num_cards + 1);
        total_str_bytes += strlen(s) + 1;
        num_cards++;
    }

    /* 5. Write binary NGLIB001 blob atomically */
    char *tmp_path = tprintf("%s.tmp.%d", out_file, (int)getpid());
    FILE *fo = fopen(tmp_path, "wb");
    if (!fo) {
        fprintf(cp_err, "Error: precompile cannot create '%s'\n", tmp_path);
        tfree(tmp_path);
        nglib_free_cards(top_head);
        nglib_free_cards(synth_head);
        nglib_free_cards(sub_head);
        nglib_free_cards(out_head);
        for (int i = 0; i < nzeros; i++) tfree(zeros[i]);
        FREE(zeros);
        return 1;
    }

    NglibHeader hdr;
    memcpy(hdr.magic, NGLIB_MAGIC, NGLIB_MAGIC_LEN);
    hdr.version = NGLIB_VERSION;
    hdr.compat_mask = nglib_current_compat_mask();
    hdr.num_cards = num_cards;
    hdr.total_str_bytes = total_str_bytes;

    uint32_t *lens = TMALLOC(uint32_t, num_cards > 0 ? num_cards : 1);
    uint32_t idx = 0;
    for (struct card *c = out_head; c; c = c->nextcard)
        lens[idx++] = (uint32_t)strlen(c->line);

    int write_ok = 1;
    if (fwrite(&hdr, sizeof(hdr), 1, fo) != 1) write_ok = 0;
    if (write_ok && num_cards > 0 && fwrite(lens, sizeof(uint32_t), num_cards, fo) != num_cards)
        write_ok = 0;
    for (struct card *c = out_head; write_ok && c; c = c->nextcard) {
        size_t sl = strlen(c->line) + 1;
        if (fwrite(c->line, 1, sl, fo) != sl)
            write_ok = 0;
    }
    if (fclose(fo) != 0) write_ok = 0;
    FREE(lens);

    if (!write_ok || rename(tmp_path, out_file) != 0) {
        fprintf(cp_err, "Error: precompile failed writing '%s'\n", out_file);
        remove(tmp_path);
        tfree(tmp_path);
        nglib_free_cards(top_head);
        nglib_free_cards(synth_head);
        nglib_free_cards(sub_head);
        nglib_free_cards(out_head);
        for (int i = 0; i < nzeros; i++) tfree(zeros[i]);
        FREE(zeros);
        return 1;
    }
    tfree(tmp_path);

    double dt_ms = nglib_now_ms() - t0;
    fprintf(stdout,
            "Precompiled library '%s' -> '%s': %u cards (%d models, %d subckts), "
            "folded %d zero-switch exprs, eliminated %d dead local params, %.1f KiB in %.1f ms\n",
            in_file, out_file, num_cards, n_models, n_subckts,
            folded_exprs, dead_params,
            (double)(sizeof(hdr) + (size_t)num_cards * 4 + total_str_bytes) / 1024.0,
            dt_ms);

    nglib_free_cards(top_head);
    nglib_free_cards(synth_head);
    nglib_free_cards(sub_head);
    nglib_free_cards(out_head);
    for (int i = 0; i < nzeros; i++) tfree(zeros[i]);
    FREE(zeros);
    return 0;
}

struct card *
nglib_load_blob(const char *blob_path, int *line_num_io)
{
    FILE *fi = fopen(blob_path, "rb");
    if (!fi) {
        fprintf(cp_err, "Error: .uselib cannot open precompiled library '%s'\n", blob_path);
        return NULL;
    }

    NglibHeader hdr;
    if (fread(&hdr, sizeof(hdr), 1, fi) != 1 ||
        memcmp(hdr.magic, NGLIB_MAGIC, NGLIB_MAGIC_LEN) != 0 ||
        hdr.version != NGLIB_VERSION)
    {
        fprintf(cp_err, "Error: '%s' is not a valid NGLIB001 precompiled library\n", blob_path);
        fclose(fi);
        return NULL;
    }

    uint32_t cur_mask = nglib_current_compat_mask();
    if (hdr.compat_mask != cur_mask) {
        fprintf(cp_err,
                "Warning: .uselib '%s' compat mask (0x%x) differs from current mode (0x%x)\n",
                blob_path, hdr.compat_mask, cur_mask);
    }

    if (hdr.num_cards == 0) {
        fclose(fi);
        return NULL;
    }

    uint32_t *lens = TMALLOC(uint32_t, hdr.num_cards);
    char *payload = TMALLOC(char, (size_t)hdr.total_str_bytes + 1);
    if (fread(lens, sizeof(uint32_t), hdr.num_cards, fi) != hdr.num_cards ||
        fread(payload, 1, (size_t)hdr.total_str_bytes, fi) != (size_t)hdr.total_str_bytes)
    {
        fprintf(cp_err, "Error: truncated precompiled library '%s'\n", blob_path);
        FREE(lens);
        FREE(payload);
        fclose(fi);
        return NULL;
    }
    fclose(fi);
    payload[hdr.total_str_bytes] = '\0';

    struct card *head = NULL, *tail = NULL;
    size_t off = 0;
    int lnum = line_num_io ? *line_num_io : 1;
    for (uint32_t i = 0; i < hdr.num_cards; i++) {
        uint32_t sl = lens[i];
        if (off + sl + 1 > hdr.total_str_bytes) {
            fprintf(cp_err, "Error: corrupt precompiled library '%s'\n", blob_path);
            nglib_free_cards(head);
            FREE(lens);
            FREE(payload);
            return NULL;
        }
        struct card *c = TMALLOC(struct card, 1);
        c->linenum = lnum++;
        c->linenum_orig = (int)i + 1;
        c->linesource = NULL;
        c->line = TMALLOC(char, (size_t)sl + 1);
        memcpy(c->line, payload + off, (size_t)sl + 1);
        c->error = NULL;
        c->nextcard = NULL;
        c->actualLine = NULL;
        c->compmod = 0;
        if (tail) tail->nextcard = c;
        else head = c;
        tail = c;
        off += (size_t)sl + 1;
    }

    if (line_num_io)
        *line_num_io = lnum;

    FREE(lens);
    FREE(payload);
    return head;
}

void
com_precompile(wordlist *wl)
{
    if (!wl || !wl->wl_next) {
        fprintf(cp_err, "Usage: precompile <out.nglib> <in_pdk.spice>\n");
        return;
    }
    (void)nglib_precompile(wl->wl_next->wl_word, wl->wl_word);
}

#endif /* NGSPICE_NGLIB_CACHE_H */
