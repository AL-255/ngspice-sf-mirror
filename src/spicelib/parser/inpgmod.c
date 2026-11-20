/**********
Copyright 1990 Regents of the University of California.  All rights reserved.
Copyright 2000 The ngspice team
3-Clause BSD license
(see COPYING or https://opensource.org/licenses/BSD-3-Clause)
Author: 1985 Thomas L. Quarles, 1991 David A. Gates
Modified: 2001 Paolo Nenzi (Cider Integration)
**********/

#include "ngspice/ngspice.h"
#include "ngspice/inpdefs.h"
#include "ngspice/ifsim.h"
#include "ngspice/cpstd.h"
#include "ngspice/fteext.h"
#include "ngspice/compatmode.h"
#include "ngspice/devdefs.h"
#include "inpxx.h"
#include <errno.h>
#include <stdio.h>
#include <string.h>

#ifdef CIDER

#include "ngspice/numcards.h"
#include "ngspice/carddefs.h"
#include "ngspice/numgen.h"
#include "ngspice/suffix.h"

#define E_MISSING    -1
#define E_AMBIGUOUS  -2

extern IFcardInfo *INPcardTab[];
extern int INPnumCards;

static int INPparseNumMod(CKTcircuit *ckt, INPmodel *model, INPtables *tab, char **errMessage);
static int INPfindCard(char *name, IFcardInfo *table[], int numCards);
static int INPfindParm(char *name, IFparm *table, int numParms);

#endif

extern INPmodel *modtab;
extern NGHASHPTR modtabhash;


static IFparm *
find_model_parameter(const char *name, IFdevice *device)
{
    IFparm *p = device->modelParms;
    IFparm *p_end = p + *(device->numModelParms);

    for (; p < p_end; p++)
        if (strcmp(name, p->keyword) == 0)
            return p;

    return NULL;
}


static IFparm *
find_instance_parameter(const char *name, IFdevice *device)
{
    IFparm *p = device->instanceParms;
    IFparm *p_end = p + *(device->numInstanceParms);

    for (; p < p_end; p++)
        if (strcmp(name, p->keyword) == 0)
            return p;

    return NULL;
}


/*
 * code moved from INPgetMod
 */
static int
create_model(CKTcircuit *ckt, INPmodel *modtmp, INPtables *tab)
{
    char    *err = NULL, *line, *parm, *endptr;
    int     error;

    /* not already defined, so create & give parameters */
    error = ft_sim->newModel(ckt, modtmp->INPmodType, &(modtmp->INPmodfast), modtmp->INPmodName);
    if (error)
        return error;

#ifdef CIDER
    /* Handle Numerical Models Differently */
    if (modtmp->INPmodType == INPtypelook("NUMD") ||
        modtmp->INPmodType == INPtypelook("NBJT") ||
        modtmp->INPmodType == INPtypelook("NUMD2") ||
        modtmp->INPmodType == INPtypelook("NBJT2") ||
        modtmp->INPmodType == INPtypelook("NUMOS"))
    {
        error = INPparseNumMod(ckt, modtmp, tab, &err);
        if (error)
            return error;
        modtmp->INPmodLine->error = err;
        return 0;
    }
#endif

    IFdevice *device = ft_sim->devices[modtmp->INPmodType];

    /* parameter isolation, identification, binding */

    line = modtmp->INPmodLine->line;

#ifdef TRACE
    printf("In INPgetMod, inserting new model into table.  line = %s ...\n", line);
#endif

    INPgetTok(&line, &parm, 1);        /* throw away '.model' */
    tfree(parm);
    INPgetNetTok(&line, &parm, 1);        /* throw away 'modname' */
    tfree(parm);

#ifdef OSDI
    /* osdi models don't accept their device type as an argument */
    bool is_osdi = false;
    if (device->registry_entry){ 
        INPgetNetTok(&line, &parm, 1); /* throw away osdi */
        tfree(parm);
        is_osdi = true;
    }
#endif

    while (*line) {
        INPgetTok(&line, &parm, 1);
        if (!*parm) {
            FREE(parm);
            continue;
        }

        IFparm *p = find_model_parameter(parm, device);

        if (p) {
#ifdef OSDI
            if (is_osdi && (p->dataType & IF_VECTOR)){  
                // we need to get rid if the leading [ in order to make sure 
                // that INPgetValue can parse the value properly
                // This is because, unlike other SPICEDev, OSDI models receive
                // array params in the syntax (param_name=[...])
                ++line;
            }
#endif
            IFvalue *val = INPgetValue(ckt, &line, p->dataType, tab);
            error = ft_sim->setModelParm(ckt, modtmp->INPmodfast, p->id, val, NULL);
            if (error)
                return error;
        } else if ((strcmp(parm, "level") == 0) || (strcmp(parm, "m") == 0)) {
            /* no instance parameter default for level and multiplier */
            /* just grab the number and throw away */
            /* since we already have that info from pass1 */
            INPgetValue(ckt, &line, IF_REAL, tab);
        
        } else {

            p = find_instance_parameter(parm, device);

            if (p) {
                char *value;

                INPgetTok(&line, &value, 1);
                if (p->dataType & IF_SET) {
                    modtmp->INPmodfast->defaults =
                        wl_cons(copy(parm),
                                wl_cons(value, modtmp->INPmodfast->defaults));
                } else {
                    fprintf(stderr,
                            "Ignoring attempt to set a default "
                            "for read-only instance parameter %s in:\n  %s\n",
                            p->keyword, modtmp->INPmodLine->line);
                }
            } else {

                double dval;

                /* want only the parameter names in output - not the values */
                errno = 0;    /* To distinguish success/failure after call */
                dval = strtod(parm, &endptr);
                /* Check for various possible errors */
                if ((errno == ERANGE && dval == HUGE_VAL) || errno != 0) {
                    perror("strtod");
                    controlled_exit(EXIT_FAILURE);
                }
                if (endptr == parm) /* it was no number - it is really a string */
                    err = INPerrCat(err,
                                    tprintf("unrecognized parameter (%s) - ignored",
                                            parm));
            }
        }
        FREE(parm);
    }

    modtmp->INPmodLine->error = err;
    return 0;
}


static void
scan_net_tok(char **line, const char **tok_start, size_t *tok_len)
{
    char *point = *line;
    char separator;
    int signstate;

    for (; *point != '\0'; point++) {
        char c = *point;
        if (c == ' ' || c == '\t' || c == '=' || c == '(' || c == ')' || c == ',')
            continue;
        break;
    }

    if (*point == '"' || *point == '\'') {
        separator = *point++;
    } else {
        separator = '\0';
    }

    *line = point;
    signstate = 0;
    for (; *point != '\0'; point++) {
        char c = *point;
        if (separator) {
            if (c == separator)
                break;
            continue;
        }
        if (c == ' ' || c == '\t' || c == '=' || c == '(' || c == ')' || c == ',' ||
            c == '*' || c == '/' || c == '^')
            break;
        if (c == '+' || c == '-') {
            if (signstate == 1 || signstate == 3)
                break;
            signstate += 1;
            continue;
        }
        if (isdigit_c(c) || c == '.') {
            signstate = (signstate > 1) ? 3 : 1;
        } else if (tolower_c(c) == 'e' && signstate == 1) {
            signstate = 2;
        } else {
            signstate = 3;
        }
    }

    if (separator && *point == separator)
        point--;
    if (point == *line && *point)
        point++;

    *tok_start = *line;
    *tok_len = (size_t)(point - *line);

    for (; *point != '\0'; point++) {
        char c = *point;
        if (c == separator || c == ' ' || c == '\t' || c == '=' || c == ',')
            continue;
        break;
    }
    *line = point;
}


static bool
parse_line(char *line, char *tokens[], int num_tokens, double values[], bool found[])
{
    int get_index = -1;
    int i;
    int nfound = 0;
    size_t tok_lens[8];

    for (i = 0; i < num_tokens; i++) {
        found[i] = FALSE;
        tok_lens[i] = strlen(tokens[i]);
    }

    while (*line) {
        if (get_index != -1) {
            int error;
            values[get_index] = INPevaluate(&line, &error, 1);
            if (!found[get_index]) {
                found[get_index] = TRUE;
                if (++nfound == num_tokens)
                    return TRUE;
            }
            get_index = -1;
            continue;
        }

        const char *ts = NULL;
        size_t tl = 0;
        scan_net_tok(&line, &ts, &tl);

        for (i = 0; i < num_tokens; i++) {
            if (tl == tok_lens[i] && memcmp(tokens[i], ts, tl) == 0) {
                get_index = i;
                break;
            }
        }
    }

    return (nfound == num_tokens) ? TRUE : FALSE;
}


typedef struct {
    INPmodel *mod;
    double lmin, lmax, wmin, wmax;
    int illegal;
} INPModBinEntry;

typedef struct {
    INPModBinEntry *entries;
    int count;
    int cap;
} INPModBinList;

static NGHASHPTR modbin_hash = NULL;
static unsigned long modbin_gen = (unsigned long)-1;

static void
free_modbin_list(void *ptr)
{
    INPModBinList *bl = (INPModBinList *)ptr;
    if (bl) {
        FREE(bl->entries);
        FREE(bl);
    }
}

void
INPkillModBinCache(void)
{
    if (modbin_hash) {
        nghash_free(modbin_hash, free_modbin_list, NULL);
        modbin_hash = NULL;
    }
    modbin_gen = (unsigned long)-1;
}

static bool
is_binnable_type(int modtype)
{
    static int init = 0;
    static int types[11];
    if (!init) {
        types[0]  = INPtypelook("BSIM3");
        types[1]  = INPtypelook("BSIM3v32");
        types[2]  = INPtypelook("BSIM3v0");
        types[3]  = INPtypelook("BSIM3v1");
        types[4]  = INPtypelook("BSIM4");
        types[5]  = INPtypelook("BSIM4v5");
        types[6]  = INPtypelook("BSIM4v6");
        types[7]  = INPtypelook("BSIM4v7");
        types[8]  = INPtypelook("HiSIM2");
        types[9]  = INPtypelook("HiSIMHV1");
        types[10] = INPtypelook("HiSIMHV2");
        init = 1;
    }
    for (int i = 0; i < 11; i++)
        if (modtype == types[i])
            return TRUE;
    return FALSE;
}

static void
ensure_modbin_cache(void)
{
    if (modbin_hash && modbin_gen == modtab_gen)
        return;

    INPkillModBinCache();
    modbin_hash = nghash_init(NGHASH_MIN_SIZE);
    nghash_unique(modbin_hash, TRUE);
    modbin_gen = modtab_gen;

    static char *model_tokens[] = { "lmin", "lmax", "wmin", "wmax" };
    double parse_values[4];
    bool parse_found[4];

    for (INPmodel *modtmp = modtab; modtmp; modtmp = modtmp->INPnextModel) {
        const char *mname = modtmp->INPmodName;
        if (!mname)
            continue;
        const char *dot = strrchr(mname, '.');
        if (!dot || dot[1] == '\0')
            continue;
        bool all_digits = TRUE;
        for (const char *p = dot + 1; *p; p++) {
            if (!isdigit_c(*p)) {
                all_digits = FALSE;
                break;
            }
        }
        if (!all_digits)
            continue;

        if (!is_binnable_type(modtmp->INPmodType) && modtmp->INPmodType >= 0)
            continue;

        int illegal = (modtmp->INPmodType < 0) ? 1 : 0;
        if (!illegal) {
            if (!modtmp->INPmodLine || !modtmp->INPmodLine->line)
                continue;
            if (!parse_line(modtmp->INPmodLine->line, model_tokens, 4, parse_values, parse_found))
                continue;
        }

        size_t base_len = (size_t)(dot - mname);
        char base_buf[256];
        char *base_key = (base_len < sizeof(base_buf)) ? base_buf : TMALLOC(char, base_len + 1);
        memcpy(base_key, mname, base_len);
        base_key[base_len] = '\0';

        INPModBinList *bl = (INPModBinList *)nghash_find(modbin_hash, base_key);
        if (!bl) {
            bl = TMALLOC(INPModBinList, 1);
            bl->entries = NULL;
            bl->count = 0;
            bl->cap = 0;
            nghash_insert(modbin_hash, base_key, bl);
        }
        if (base_key != base_buf)
            FREE(base_key);

        if (bl->count == bl->cap) {
            int ncap = bl->cap ? bl->cap * 2 : 16;
            bl->entries = TREALLOC(INPModBinEntry, bl->entries, ncap);
            bl->cap = ncap;
        }
        INPModBinEntry *e = &bl->entries[bl->count++];
        e->mod = modtmp;
        e->illegal = illegal;
        if (!illegal) {
            e->lmin = parse_values[0];
            e->lmax = parse_values[1];
            e->wmin = parse_values[2];
            e->wmax = parse_values[3];
        } else {
            e->lmin = e->lmax = e->wmin = e->wmax = 0.0;
        }
    }
}


static bool
is_equal(double result, double expectedResult)
{
    return fabs(result - expectedResult) < 1e-9;
}


static bool
in_range(double value, double min, double max)
{
    /* the standard binning rule is: min <= value < max */
    return is_equal(value, min)  || is_equal(value, max) || (min < value && value < max);
}


char *
INPgetModBin(CKTcircuit *ckt, char *name, INPmodel **model, INPtables *tab, char *line)
{
    double       l, w;
    double       parse_values[4] = { 0.0, 0.0, 1.0, 0.0 };
    bool         parse_found[4]  = { FALSE, FALSE, FALSE, FALSE };
    static char *instance_tokens[] = { "l", "w", "nf", "wnflag" };
    double       scale;
    int          wnflag;

    if (!cp_getvar("scale", CP_REAL, &scale, 0))
        scale = 1;

    if (!cp_getvar("wnflag", CP_NUM, &wnflag, 0)) {
        if (newcompat.spe || newcompat.hs)
            wnflag = 1;
        else
            wnflag = 0;
    }

    *model = NULL;

    /* Single-pass extraction of l, w, nf, wnflag from instance line */
    (void)parse_line(line, instance_tokens, 4, parse_values, parse_found);
    if (!parse_found[0] || !parse_found[1])
        return NULL;

    if (!parse_found[2]) {
        parse_values[2] = 1.;
    } else if (parse_found[3]) {
        if (parse_values[3] == 0)
            parse_values[2] = 1.;
    } else {
        if (wnflag == 0)
            parse_values[2] = 1.;
    }

    l = parse_values[0] * scale;
    w = parse_values[1] / parse_values[2] * scale;

#if NGSPICE_OPT_GENERAL
    ensure_modbin_cache();

    INPModBinList *bl = modbin_hash ? (INPModBinList *)nghash_find(modbin_hash, name) : NULL;
    if (!bl)
        return NULL;

    for (int bi = 0; bi < bl->count; bi++) {
        INPModBinEntry *e = &bl->entries[bi];
        if (e->illegal) {
            *model = NULL;
            return tprintf("Unknown device type for model %s\n", name);
        }
        if (in_range(l, e->lmin, e->lmax) && in_range(w, e->wmin, e->wmax)) {
            INPmodel *modtmp = e->mod;
            /* create unless model is already defined */
            if (!modtmp->INPmodfast) {
                int error = create_model(ckt, modtmp, tab);
                if (error)
                    return NULL;
            }
            *model = modtmp;
            return NULL;
        }
    }
#else
    {
        static char *model_tokens[] = { "lmin", "lmax", "wmin", "wmax" };
        INPmodel *modtmp;
        size_t namelen = strlen(name);
        for (modtmp = modtab; modtmp != NULL; modtmp = modtmp->INPnextModel) {
            if (strncmp(modtmp->INPmodName, name, namelen) == 0 &&
                modtmp->INPmodName[namelen] == '.')
            {
                if (modtmp->INPmodType < 0) {
                    *model = NULL;
                    return tprintf("Unknown device type for model %s\n", name);
                }
                if (parse_line(modtmp->INPmodLine->line, model_tokens, 4, parse_values, parse_found)) {
                    if (in_range(l, parse_values[0], parse_values[1]) &&
                        in_range(w, parse_values[2], parse_values[3]))
                    {
                        if (!modtmp->INPmodfast) {
                            int error = create_model(ckt, modtmp, tab);
                            if (error)
                                return NULL;
                        }
                        *model = modtmp;
                        return NULL;
                    }
                }
            }
        }
    }
#endif

    return NULL;
}


char *
INPgetMod(CKTcircuit *ckt, char *name, INPmodel **model, INPtables *tab)
{
    INPmodel *modtmp;

#ifdef TRACE
    printf("In INPgetMod, examining model %s ...\n", name);
#endif

    if (modtabhash) {
        modtmp = nghash_find(modtabhash, name);
        if (modtmp) {
            /* found the model in question - now instantiate if necessary */
            /* and return an appropriate pointer to it */

    /* if illegal device type */
            if (modtmp->INPmodType < 0) {
#ifdef TRACE
                printf("In INPgetMod, illegal device type for model %s ...\n", name);
#endif
                * model = NULL;
                return tprintf("Unknown device type for model %s\n", name);
            }

            /* create unless model is already defined */
            if (!modtmp->INPmodfast) {
                int error = create_model(ckt, modtmp, tab);
                if (error) {
                    *model = NULL;
                    return INPerror(error);
                }
            }

            *model = modtmp;
            return NULL;
        }
    }
#if (0)
    for (modtmp = modtab; modtmp; modtmp = modtmp->INPnextModel) {

#ifdef TRACE
        printf("In INPgetMod, comparing %s against stored model %s ...\n", name, modtmp->INPmodName);
#endif

        if (strcmp(modtmp->INPmodName, name) == 0) {
            /* found the model in question - now instantiate if necessary */
            /* and return an appropriate pointer to it */

            /* if illegal device type */
            if (modtmp->INPmodType < 0) {
#ifdef TRACE
                printf("In INPgetMod, illegal device type for model %s ...\n", name);
#endif
                *model = NULL;
                return tprintf("Unknown device type for model %s\n", name);
            }

            /* create unless model is already defined */
            if (!modtmp->INPmodfast) {
                int error = create_model(ckt, modtmp, tab);
                if (error) {
                    *model = NULL;
                    return INPerror(error);
                }
            }

            *model = modtmp;
            return NULL;
        }
    }
#endif
#ifdef TRACE
    printf("In INPgetMod, didn't find model for %s, using default ...\n", name);
#endif

    *model = NULL;
    return tprintf("Unable to find definition of model %s\n", name);
}


#ifdef CIDER
/*
 * Parse a numerical model by running through the list of original
 * input cards which make up the model
 * Given:
 * 1. First card looks like: .model modname modtype <level=val>
 * 2. Other cards look like: +<whitespace>? where ? tells us what
 * to do with the next card:
 *    '#$*' = comment card
 *    '+'   = continue previous card
 *    other = new card
 */
static int
INPparseNumMod(CKTcircuit *ckt, INPmodel *model, INPtables *tab, char **errMessage)
{
    struct card *txtCard;    /* Text description of a card */
    GENcard *tmpCard = NULL; /* Processed description of a card */
    IFcardInfo *info = NULL; /* Info about the type of card located */
    char *cardName = NULL;   /* name of a card */
    int cardNum = 0;         /* number of this card in the overall line */
    char *err = NULL;        /* Strings for error messages */
    int error;

    /* Chase down to the top of the list of actual cards */
    txtCard = model->INPmodLine->actualLine;

    /* Skip the first card if it exists since there's nothing interesting */
    /* txtCard will be empty if the numerical model is empty */
    if (txtCard)
        txtCard = txtCard->nextcard;

    /* Now parse each remaining card */
    for (; txtCard; txtCard = txtCard->nextcard) {
        char *line = txtCard->line;
        cardNum++;

        /* Skip the initial '+' and any whitespace. */
        line++;
        while (*line == ' ' || *line == '\t')
            line++;

        switch (*line) {
        case '*':
        case '$':
        case '#':
        case '\0':
        case '\n':
            /* comment or empty cards */
            info = NULL;
            continue;
        case '+':
            /* continuation card */
            if (!info) {
                err = INPerrCat(err,
                                tprintf("Error on card %d : illegal continuation \'+\' - ignored",
                                        cardNum));
                continue;
            }
            /* Skip leading '+'s */
            while (*line == '+')
                line++;
            break;
        default:
            info = NULL;
            break;
        }

        if (!info) {
            /* new command card */
            if (cardName)       /* get rid of old card name */
                FREE(cardName);
            INPgetTok(&line, &cardName, 1);        /* get new card name */
            if (*cardName) {                 /* Found a name? */
                int lastType = INPfindCard(cardName, INPcardTab, INPnumCards);
                if (lastType >= 0) {
                    /* Add card structure to model */
                    info = INPcardTab[lastType];
                    error = info->newCard(&tmpCard, model->INPmodfast);
                    if (error) {
                        FREE(cardName);
                        return error;
                    }
                    /* Handle parameter-less cards */
                } else if (cinprefix(cardName, "title", 3)) {
                    /* Do nothing */
                } else if (cinprefix(cardName, "comment", 3)) {
                    /* Do nothing */
                } else if (cinprefix(cardName, "end", 3)) {
                    /* Terminate parsing */
                    *errMessage = err;
                    FREE(cardName);
                    return 0;
                } else {
                    /* Error */
                    err = INPerrCat(err,
                                    tprintf("Error on card %d : unrecognized name (%s) - ignored",
                                            cardNum, cardName));
                }
                FREE(cardName);
            }
        }

        if (!info)
            continue;

        /* parse the rest of this line */
        while (*line) {

            int invert = FALSE;
            /* Strip leading carat from booleans */
            if (*line == '^') {
                invert = TRUE;
                line++;
            }

            char *parm;                /* name of a parameter */
            INPgetTok(&line, &parm, 1);
            if (!*parm) {
                FREE(parm);
                break;
            }

            int idx = INPfindParm(parm, info->cardParms, info->numParms);
            if (idx == E_MISSING) {
                /* parm not found */
                err = INPerrCat(err,
                                tprintf("Error on card %d : unrecognized parameter (%s) - ignored",
                                        cardNum, parm));
            } else if (idx == E_AMBIGUOUS) {
                /* parm ambiguous */
                err = INPerrCat(err,
                                tprintf("Error on card %d : ambiguous parameter (%s) - ignored",
                                        cardNum, parm));
            } else {
                IFvalue *value = INPgetValue(ckt, &line, info->cardParms[idx].dataType, tab);

                /* invert if this is a boolean entry */
                if (invert) {
                    if ((info->cardParms[idx].dataType & IF_VARTYPES) == IF_FLAG)
                        value->iValue = 0;
                    else
                        err = INPerrCat(err,
                                        tprintf("Error on card %d : non-boolean parameter (%s) - \'^\' ignored",
                                                cardNum, parm));
                }

                error = info->setCardParm(info->cardParms[idx].id, value, tmpCard);
                if (info->cardParms[idx].dataType & IF_STRING) {
                    FREE(value->sValue);
                } else if (info->cardParms[idx].dataType & IF_REALVEC) {
                    FREE(value->v.vec.rVec);
                } else if (info->cardParms[idx].dataType & IF_INTVEC) {
                    FREE(value->v.vec.iVec);
                }
                if (error)
                    return error;
            }
            FREE(parm);
        }
    }

    *errMessage = err;
    return 0;
}


/*
 * Locate the best match to a card name in an IFcardInfo table
 */
static int
INPfindCard(char *name, IFcardInfo *table[], int numCards)
{
    int length = (int) strlen(name);
    int best = E_MISSING;
    int bestMatch = 0;

    int test;

    /* compare all the names in the card table to this name */
    for (test = 0; test < numCards; test++) {
        int match = cimatch(name, table[test]->name);
        if ((match > 0) && (match == bestMatch)) {
            best = E_AMBIGUOUS;
        } else if ((match > bestMatch) && (match == length)) {
            best = test;
            bestMatch = match;
        }
    }

    return best;
}


/*
 * Locate the best match to a parameter name in an IFparm table
 */
static int
INPfindParm(char *name, IFparm *table, int numParms)
{
    int length = (int) strlen(name);
    int best = E_MISSING;
    int bestMatch = 0;
    int bestId = -1;

    int test;

    /* compare all the names in the parameter table to this name */
    for (test = 0; test < numParms; test++) {
        int match = cimatch(name, table[test].keyword);
        if ((match == length) && (match == (int) strlen(table[test].keyword))) {
            /* exact match */
            return test;
        }
        int id = table[test].id;
        if ((match > 0) && (match == bestMatch) && (id != bestId)) {
            best = E_AMBIGUOUS;
        } else if ((match > bestMatch) && (match == length)) {
            bestMatch = match;
            bestId = id;
            best = test;
        }
    }

    return best;
}

#endif /* CIDER */
