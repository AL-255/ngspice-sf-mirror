#ifndef NGSPICE_INP_SPECTRE_H
#define NGSPICE_INP_SPECTRE_H

#include <stdio.h>
#include <stddef.h>
#include "ngspice/bool.h"

/*
 * Native Spectre (.scs) model-file & netlist compatibility frontend.
 *
 * Triggered when `set ngbehavior=spe` (or `spea`) is enabled, or automatically
 * when an included/library file has a `.scs` extension or starts with
 * `simulator lang=spectre`.
 *
 * Translates Spectre syntax into standard ngspice SPICE cards on the fly so
 * commercial Spectre PDK files (sections, parameters, binned/unbinned models,
 * inline/regular subckts, statistics blocks, Spectre device instances, and
 * `ahdl_include` Verilog-A / OSDI modules) can be loaded directly by ngspice.
 */

/* Return TRUE if `fp` (at its current offset) should be translated as Spectre.
 * Always restores `fp` to its original offset before returning. */
bool inp_spectre_should_translate(FILE *fp, const char *file_name);

/* Translate the entire Spectre stream `fp` into a malloc'd array of malloc'd
 * SPICE lines (each NUL-terminated with trailing '\n', ready for inp_read()).
 * When `is_top_level` is TRUE (call_depth == 0), emits a SPICE title line first.
 * Sets `*out_nlines` to the number of lines returned. Returns NULL on error. */
char **inp_spectre_translate_fp(FILE *fp, const char *dir_name,
                                const char *file_name, bool is_top_level,
                                size_t *out_nlines);

#endif /* NGSPICE_INP_SPECTRE_H */
