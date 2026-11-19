/*
 * Apple Silicon (ARM64 NEON) Architecture-Specific BSIM4v5 & BSIM4v6 Core Kernel.
 *
 * Tuned from Apple M6 profiling & microarchitecture measurements:
 *   - Default B4A_W=4 (2x float64x2_t) saturates M6's 4x 128-bit FP64 pipes on BSIM4's
 *     deep serial dependency chain (2x higher ILP than W=2), while splitting into
 *     3 __attribute__((noinline)) phases cuts spills & L1I footprint vs monolithic W=4.
 *   - Shared scalar BSIM4ApplePParam (~1 KiB per unique (model, pParam), 54-160 KiB total)
 *     fits in Apple M6's 128 KiB L1D with ARM64 ld1r broadcasts and scalar feature branches.
 *   - Single compiled kernel shared by both BSIM4v5 and BSIM4v6 (2x smaller L1I).
 *   - Exact 128B BSIM4AppleCheck + pack-level step validation (L1 hits touch ONLY 1 cache line!).
 *   - Per-instance (W=1) stationary cache + dynamic compaction into dense B4A_W-wide batches
 *     (-18% to -23% fewer full physics evals vs static W=4 blocks).
 *   - Adaptive incremental KLU delta-stamping (0 matrix/RHS stamps on L1 hits!).
 */
#ifndef NGSPICE_BSIM4_APPLE_CORE_H
#define NGSPICE_BSIM4_APPLE_CORE_H

#include "ngspice/opt_config.h"

#if NGSPICE_OPT_APPLE

#include <arm_neon.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef B4A_W
#define B4A_W 4
#endif

#define B4V_MAX_EXP        5.834617425e14
#define B4V_MIN_EXP        1.713908431e-15
#define B4V_EXP_THRESHOLD  34.0
#define B4V_LOG1P_MIN_EXP  1.713908431e-15
#define B4V_DELTA_1        0.02
#define B4V_DELTA_3        0.02
#define B4V_DELTA_4        0.02
#define B4V_EPSSI          1.03594e-10

#define B4A_JF_SRC_SAT   (1u << 0)
#define B4A_JF_DRN_SAT   (1u << 1)
#define B4A_JF_CZBS      (1u << 2)
#define B4A_JF_CZBSSW    (1u << 3)
#define B4A_JF_CZBSSWG   (1u << 4)
#define B4A_JF_CZBD      (1u << 5)
#define B4A_JF_CZBDSW    (1u << 6)
#define B4A_JF_CZBDSWG   (1u << 7)
#define B4A_JF_TT_S      (1u << 8)
#define B4A_JF_TT_D      (1u << 9)
#define B4A_JF_TT_SSW    (1u << 10)
#define B4A_JF_TT_DSW    (1u << 11)
#define B4A_JF_TT_SSWG   (1u << 12)
#define B4A_JF_TT_DSWG   (1u << 13)

/* ========================================================================= */
/* Inline Apple Silicon NEON exp (<=2 ULP) & 128-entry L1-table log (<=3 ULP)*/
/* ========================================================================= */
#define B4V_CHARGE_Q       1.60219e-19
#define B4V_DELTA_1        0.02
#define B4V_DELTA_3        0.02
#define B4V_DELTA_4        0.02

typedef struct { double invc; double hi; double lo; } b4a_log_entry_t;
static const b4a_log_entry_t b4a_log_tab[128] = {
    { 0x1.6a133d0d9005fp+0, -0x1.62fe995db5112p-2, 0x1.335c38775ca3bp-56 },
    { 0x1.6815f2f480f0ep+0, -0x1.5d5a48dc947f0p-2, 0x1.12a6693705c41p-57 },
    { 0x1.661e39bef312ap+0, -0x1.57bde25a3ce79p-2, 0x1.eb5b78ad06ac4p-56 },
    { 0x1.642bfa3192f7cp+0, -0x1.52294fc1c2605p-2, 0x1.3331985029bb4p-57 },
    { 0x1.623f1d91a43b9p+0, -0x1.4c9c7b5a23f2dp-2, -0x1.4aa5dbc9283bap-56 },
    { 0x1.60578da18c9dbp+0, -0x1.47174fc45033dp-2, 0x1.f8d64bb78259bp-56 },
    { 0x1.5e75349d7b7c3p+0, -0x1.4199b7f93774ap-2, 0x1.9b04661f6241dp-57 },
    { 0x1.5c97fd382ca3bp+0, -0x1.3c239f47eb208p-2, -0x1.dcfae71b91751p-56 },
    { 0x1.5abfd297c5723p+0, -0x1.36b4f153c9d90p-2, -0x1.b93077c7df86bp-59 },
    { 0x1.58eca052cb548p+0, -0x1.314d9a12b7eb2p-2, -0x1.a4c3492f9580dp-56 },
    { 0x1.571e526d32affp+0, -0x1.2bed85cb63b8fp-2, 0x1.f491d83ecc97dp-57 },
    { 0x1.5554d55585554p+0, -0x1.2694a11395b91p-2, 0x1.1f150e2686307p-56 },
    { 0x1.539015e21fa11p+0, -0x1.2142d8ce8badap-2, 0x1.b8ec324944119p-58 },
    { 0x1.51d0014e83795p+0, -0x1.1bf81a2b5ebb4p-2, -0x1.3e632ba0a9917p-60 },
    { 0x1.50148538c05e9p+0, -0x1.16b452a3740a4p-2, 0x1.542c908f3994dp-57 },
    { 0x1.4e5d8f9eefd01p+0, -0x1.11776ff8f79dbp-2, -0x1.c3c0f864169e5p-56 },
    { 0x1.4cab0edcc549fp+0, -0x1.0c4160356111bp-2, 0x1.018a609bb4b1ap-56 },
    { 0x1.4afcf1a9312c8p+0, -0x1.071211a801f42p-2, 0x1.a5909a5e027a8p-58 },
    { 0x1.4953271415e19p+0, -0x1.01e972e49d6c7p-2, -0x1.911e32aa8f449p-56 },
    { 0x1.47ad9e840e9d4p+0, -0x1.f98ee58411d8dp-3, 0x1.4e558077950ccp-57 },
    { 0x1.460c47b4471c6p+0, -0x1.ef5800b1ab443p-3, -0x1.807132b47ab0dp-65 },
    { 0x1.446f12b263ca9p+0, -0x1.e52e160404d7ap-3, 0x1.5e16fb20c39dcp-57 },
    { 0x1.42d5efdc79be7p+0, -0x1.db1104ab6c133p-3, -0x1.736a7ed685f20p-57 },
    { 0x1.4140cfdf16019p+0, -0x1.d100ac5402a35p-3, 0x1.891c4c7e3ce3cp-57 },
    { 0x1.3fafa3b3539e1p+0, -0x1.c6fced2352531p-3, 0x1.115d60233d22ep-57 },
    { 0x1.3e225c9cfff0ep+0, -0x1.bd05a7b5f0121p-3, 0x1.38dd4dd2d6f87p-58 },
    { 0x1.3c98ec28ccc68p+0, -0x1.b31abd1d2da52p-3, 0x1.b9cdb7ef665afp-57 },
    { 0x1.3b13442a8fc9ep+0, -0x1.a93c0edcd98bep-3, -0x1.6d7f907628c83p-60 },
    { 0x1.399156bb8ed43p+0, -0x1.9f697ee90cb97p-3, -0x1.8f6b0a2a61dcdp-61 },
    { 0x1.38131638d8af9p+0, -0x1.95a2efa405b90p-3, 0x1.41ca9691733b0p-59 },
    { 0x1.36987541a9e25p+0, -0x1.8be843dc10d95p-3, -0x1.2b64743b71f70p-58 },
    { 0x1.352166b5dd1e0p+0, -0x1.82395ec97d067p-3, -0x1.ad2c92aa02dc0p-58 },
    { 0x1.33adddb466ef8p+0, -0x1.7896240c9cf03p-3, -0x1.65c432a92c396p-57 },
    { 0x1.323dcd99dc523p+0, -0x1.6efe77abd4278p-3, 0x1.e22d34c530e1cp-57 },
    { 0x1.30d129ff03cbfp+0, -0x1.65723e11afdf5p-3, -0x1.7c55448a76090p-57 },
    { 0x1.2f67e6b770b96p+0, -0x1.5bf15c0b0af78p-3, 0x1.1426850abacdap-58 },
    { 0x1.2e01f7d02877bp+0, -0x1.527bb6c53d096p-3, 0x1.f44cff5d050d0p-58 },
    { 0x1.2c9f518e5119dp+0, -0x1.491133cc54230p-3, 0x1.ad3286f84c3c4p-57 },
    { 0x1.2b3fe86de95b4p+0, -0x1.3fb1b90958e6dp-3, -0x1.0ed0c2ac8ac68p-57 },
    { 0x1.29e3b1208985ap+0, -0x1.365d2cc09ccabp-3, 0x1.d3a99e268cd84p-58 },
    { 0x1.288aa08c2cff8p+0, -0x1.2d13759012285p-3, 0x1.21df1dce17fa8p-57 },
    { 0x1.2734abca043fap+0, -0x1.23d47a6dade3cp-3, 0x1.0385f4828b365p-57 },
    { 0x1.25e1c8254ee07p+0, -0x1.1aa022a5d25fap-3, -0x1.27683daf84077p-57 },
    { 0x1.2491eb1a3d930p+0, -0x1.117655d9c3813p-3, 0x1.84f7dfecc8f5bp-57 },
    { 0x1.23450a54dbb2dp+0, -0x1.0856fbfe23865p-3, -0x1.b0e9aebea302ep-63 },
    { 0x1.21fb1bb0003dcp+0, -0x1.fe83fab2f0e1ep-4, -0x1.1d00215e488bfp-61 },
    { 0x1.20b4153445f63p+0, -0x1.ec6e8505739dep-4, -0x1.ce869cc60d22cp-60 },
    { 0x1.1f6fed170a768p+0, -0x1.da6d68bfcd479p-4, 0x1.98482805bba31p-58 },
    { 0x1.1e2e99b973ffdp+0, -0x1.c8807849467bfp-4, 0x1.0ab8a8f03315fp-58 },
    { 0x1.1cf011a77dd00p+0, -0x1.b6a786a1bd8ddp-4, -0x1.ee2f121249df8p-59 },
    { 0x1.1bb44b970acb6p+0, -0x1.a4e2675f00a5bp-4, 0x1.f4726d5c386f5p-58 },
    { 0x1.1a7b3e66fe49fp+0, -0x1.9330eeaa3688cp-4, 0x1.7c50fe579d121p-58 },
    { 0x1.1944e11e5ad8fp+0, -0x1.8192f13d55a6cp-4, -0x1.346903dc7fe11p-59 },
    { 0x1.18112aeb66c28p+0, -0x1.70084460a9063p-4, -0x1.7e71609418193p-59 },
    { 0x1.16e01322d6302p+0, -0x1.5e90bde862c0bp-4, 0x1.381e4b52dfa36p-58 },
    { 0x1.15b1913efabd1p+0, -0x1.4d2c34323bb0cp-4, 0x1.75d2fcaa46609p-64 },
    { 0x1.14859cdef84f3p+0, -0x1.3bda7e231ff77p-4, -0x1.bbc4b06706b18p-58 },
    { 0x1.135c2dc5ff0f6p+0, -0x1.2a9b7324e8110p-4, 0x1.4f681d0bf6d4dp-59 },
    { 0x1.12353bda8a5a9p+0, -0x1.196eeb241e21ep-4, -0x1.a1f50718131b3p-59 },
    { 0x1.1110bf25a485cp+0, -0x1.0854be8dcf250p-4, -0x1.afd2bd801fcf3p-58 },
    { 0x1.0feeafd22f52cp+0, -0x1.ee998c9acf8eep-5, 0x1.c85c6dfb9dc7ap-59 },
    { 0x1.0ecf062c30f0cp+0, -0x1.ccadb795390d9p-5, 0x1.b519e5257f89fp-60 },
    { 0x1.0db1baa025695p+0, -0x1.aae5b1ceb9d72p-5, 0x1.63ad684a1d33ep-63 },
    { 0x1.0c96c5ba5457bp+0, -0x1.89412ffba3003p-5, 0x1.b7faba5a9076dp-60 },
    { 0x1.0b7e20262acc1p+0, -0x1.67bfe7bcc8a04p-5, 0x1.892d618499146p-60 },
    { 0x1.0a67c2ad993b8p+0, -0x1.46618f9ba7444p-5, -0x1.f8379413f3b02p-59 },
    { 0x1.0953a63875608p+0, -0x1.2525df069d7d4p-5, -0x1.11e1bf2044d14p-61 },
    { 0x1.0841c3cbdfee2p+0, -0x1.040c8e4d38dd2p-5, -0x1.ba7a7f5b06ea5p-60 },
    { 0x1.07321489adfb2p+0, -0x1.c62aad392c0e8p-6, -0x1.5040553bfc072p-60 },
    { 0x1.062491afd60adp+0, -0x1.847fe3f7a6d69p-6, 0x1.d1f570f50706ap-60 },
    { 0x1.05193497e0984p+0, -0x1.431836912c0bdp-6, -0x1.2c31f7a68c857p-61 },
    { 0x1.040ff6b65c0cfp+0, -0x1.01f31c673ff87p-6, 0x1.85109802659d2p-62 },
    { 0x1.0308d19a5408bp+0, -0x1.82201cf5e18cbp-7, -0x1.a9dadccdfaa4bp-61 },
    { 0x1.0203beeccbe53p+0, -0x1.00dd0ecc901e7p-7, -0x1.0aba206407750p-64 },
    { 0x1.0100b8703c5cdp+0, -0x1.00380d5774cbdp-8, -0x1.ab67aae1900f7p-62 },
    { 0x1.0000000000000p+0, 0x0.0p+0, 0x0.0p+0 },
    { 0x1.fc06d49302688p-1, 0x1.fe9188c4af936p-8, -0x1.823d77d1a30bcp-67 },
    { 0x1.f81e6ac3d49f6p-1, 0x1.fc516f6a65708p-7, 0x1.180e8bf0265c9p-63 },
    { 0x1.f44546ef6b93ep-1, 0x1.7bb4dd6602840p-6, -0x1.67785e4da0a40p-60 },
    { 0x1.f07b1038fee8bp-1, 0x1.f84c99982c747p-6, 0x1.3c9011f7113cap-60 },
    { 0x1.ecbf706ff153cp-1, 0x1.39f9ce5f90259p-5, 0x1.a46444a3962c8p-59 },
    { 0x1.e91213f649557p-1, 0x1.7756c10a7fff4p-5, 0x1.8b15700c1a4c8p-59 },
    { 0x1.e572a9a84c63ep-1, 0x1.b43ee81e1b18fp-5, -0x1.8befd31929d89p-59 },
    { 0x1.e1e0e2c52d97ap-1, 0x1.f0b3fced8b192p-5, 0x1.b3d3f5c028c63p-59 },
    { 0x1.de5c72d8c1c62p-1, 0x1.165bd78c72248p-4, -0x1.e8ef63e3466e5p-58 },
    { 0x1.dae50fa62bc70p-1, 0x1.3425d26ab1048p-4, -0x1.01e1546b52a97p-58 },
    { 0x1.d77a711374647p-1, 0x1.51b8bd9983d93p-4, -0x1.3e239ddb20905p-58 },
    { 0x1.d41c5116022f3p-1, 0x1.6f15632fe1433p-4, 0x1.732eabf74ff21p-58 },
    { 0x1.d0ca6b9fe617ap-1, 0x1.8c3c88f414055p-4, -0x1.7a392a918011bp-63 },
    { 0x1.cd847e8df24f7p-1, 0x1.a92ef07af55a0p-4, -0x1.0465a5d7d04cfp-58 },
    { 0x1.ca4a4996918cdp-1, 0x1.c5ed57460ec7ep-4, 0x1.dd9609e3a2eb9p-58 },
    { 0x1.c71b8e3955552p-1, 0x1.e27876e0af2efp-4, -0x1.ad7f354f66f72p-58 },
    { 0x1.c3f80faf32795p-1, 0x1.fed104fbfe355p-4, -0x1.00eea97c63fc5p-61 },
    { 0x1.c0df92db6369cp-1, 0x1.0d7bd9c50c5b4p-3, 0x1.3275419d8a6cfp-59 },
    { 0x1.bdd1de3ce87b7p-1, 0x1.1b76986c202abp-3, 0x1.73c08d8f9713bp-58 },
    { 0x1.baceb9e09ea78p-1, 0x1.295913d413e45p-3, -0x1.798993558372bp-57 },
    { 0x1.b7d5ef53e5b1bp-1, 0x1.37239fa4328ebp-3, 0x1.bc50905b9efafp-57 },
    { 0x1.b4e74997cf05ap-1, 0x1.44d68dd60954fp-3, -0x1.0f47d92b57c4dp-57 },
    { 0x1.b2029514cef46p-1, 0x1.52722ec0d3827p-3, -0x1.e5902a2c5107fp-57 },
    { 0x1.af279f8eea508p-1, 0x1.5ff6d12486015p-3, -0x1.0992fea33364bp-59 },
    { 0x1.ac56381a5ab91p-1, 0x1.6d64c2347e211p-3, -0x1.b5a1cda899ae5p-58 },
    { 0x1.a98e2f10a42d7p-1, 0x1.7abc4da1d74a8p-3, 0x1.8b1c12e50c330p-57 },
    { 0x1.a6cf560616cd4p-1, 0x1.87fdbda56b128p-3, 0x1.008e6b5095413p-59 },
    { 0x1.a4197fbfb7e92p-1, 0x1.95295b097efbap-3, -0x1.e1cc5bb641d9cp-57 },
    { 0x1.a16c80298dc91p-1, 0x1.a23f6d33230f7p-3, 0x1.f5ccf460b29edp-58 },
    { 0x1.9ec82c4d49c95p-1, 0x1.af403a2b44514p-3, 0x1.717a5417f8511p-58 },
    { 0x1.9c2c5a494ca67p-1, 0x1.bc2c06a775ebep-3, 0x1.c263c1b663de0p-58 },
    { 0x1.9998e14801060p-1, 0x1.c903161273dfcp-3, 0x1.d377c29714e4cp-57 },
    { 0x1.970d9977887cap-1, 0x1.d5c5aa9461d06p-3, -0x1.3116160b2c4c8p-58 },
    { 0x1.948a5c01b780dp-1, 0x1.e274051ac8775p-3, 0x1.a84471a960d84p-59 },
    { 0x1.920f03045ce68p-1, 0x1.ef0e65605428fp-3, 0x1.8d18c0fa4fe89p-58 },
    { 0x1.8f9b6989d1aa8p-1, 0x1.fb9509f456b6bp-3, -0x1.8aba8a57626f0p-57 },
    { 0x1.8d2f6b81cdfc2p-1, 0x1.040418210773bp-2, -0x1.7f9cdc00aa964p-56 },
    { 0x1.8acae5ba80998p-1, 0x1.0a340a4bdbd1ep-2, 0x1.d139319a4edc3p-56 },
    { 0x1.886db5d9e5b78p-1, 0x1.105a7916b06dfp-2, -0x1.89fe20dbc73c7p-56 },
    { 0x1.8617ba575ace6p-1, 0x1.16778195ca0d6p-2, 0x1.6d6519df8429cp-58 },
    { 0x1.83c8d2756cc66p-1, 0x1.1c8b4058e3521p-2, -0x1.dbf7c131d4525p-56 },
    { 0x1.8180de3bde1cep-1, 0x1.2295d16e4d70cp-2, 0x1.856272da10c5cp-56 },
    { 0x1.7f3fbe71e2b6ap-1, 0x1.28975065f96f3p-2, 0x1.d03e24e0cd58ap-57 },
    { 0x1.7d0554988f314p-1, 0x1.2e8fd8546ab63p-2, 0x1.79fa52599f4dap-56 },
    { 0x1.7ad182e5799d9p-1, 0x1.347f83d593c63p-2, -0x1.0fbd0f52b68f1p-56 },
    { 0x1.78a42c3d89a5dp-1, 0x1.3a666d0f9dce6p-2, 0x1.9c978af400817p-57 },
    { 0x1.767d342ff63b4p-1, 0x1.4044adb59bdf1p-2, -0x1.ce0d8ed5e1112p-57 },
    { 0x1.745c7ef16efa5p-1, 0x1.461a5f0a2a6b3p-2, 0x1.e18bee8add2fcp-56 },
    { 0x1.7241f1576f8b7p-1, 0x1.4be799e1fbc4bp-2, -0x1.c741742ff4013p-60 },
    { 0x1.702d70d3bb59dp-1, 0x1.51ac76a652365p-2, -0x1.c11f4b975dbf3p-56 },
    { 0x1.6e1ee370000c3p-1, 0x1.57690d57685b0p-2, 0x1.baf74481b2ae1p-59 },
    { 0x1.6c162fc99d3d9p-1, 0x1.5d1d758ec84aep-2, -0x1.a249a84922b42p-56 },
};

/* ========================================================================= */
/* Inline IEEE-754 double 2-wide NEON exp (poly) & log (zero-div L1 table)   */
/* ========================================================================= */
static inline float64x2_t b4a_exp2(float64x2_t x) {
    const float64x2_t inv_ln2 = vdupq_n_f64(1.44269504088896340736);
    const float64x2_t ln2_hi  = vdupq_n_f64(6.93147180369123816490e-01);
    const float64x2_t ln2_lo  = vdupq_n_f64(1.90821492927058770002e-10);

    float64x2_t n = vrndnq_f64(vmulq_f64(x, inv_ln2));
    float64x2_t r = vfmsq_f64(x, n, ln2_hi);
    r = vfmsq_f64(r, n, ln2_lo);

    float64x2_t r2 = vmulq_f64(r, r);
    float64x2_t r4 = vmulq_f64(r2, r2);
    float64x2_t r8 = vmulq_f64(r4, r4);

    float64x2_t p01 = vaddq_f64(vdupq_n_f64(1.0), r);
    float64x2_t p23 = vfmaq_f64(vdupq_n_f64(0.5), r, vdupq_n_f64(1.0 / 6.0));
    float64x2_t p45 = vfmaq_f64(vdupq_n_f64(1.0 / 24.0), r, vdupq_n_f64(1.0 / 120.0));
    float64x2_t p67 = vfmaq_f64(vdupq_n_f64(1.0 / 720.0), r, vdupq_n_f64(1.0 / 5040.0));
    float64x2_t p89 = vfmaq_f64(vdupq_n_f64(1.0 / 40320.0), r, vdupq_n_f64(1.0 / 362880.0));
    float64x2_t pAB = vfmaq_f64(vdupq_n_f64(1.0 / 3628800.0), r, vdupq_n_f64(1.0 / 39916800.0));
    float64x2_t pCD = vfmaq_f64(vdupq_n_f64(1.0 / 479001600.0), r, vdupq_n_f64(1.0 / 6227020800.0));

    float64x2_t q0 = vfmaq_f64(p01, r2, p23);
    float64x2_t q1 = vfmaq_f64(p45, r2, p67);
    float64x2_t q2 = vfmaq_f64(p89, r2, pAB);

    float64x2_t s0 = vfmaq_f64(q0, r4, q1);
    float64x2_t s1 = vfmaq_f64(q2, r4, pCD);
    float64x2_t poly = vfmaq_f64(s0, r8, s1);

    int64x2_t ki = vcvtq_s64_f64(n);
    int64x2_t exp_bits = vshlq_n_s64(vaddq_s64(ki, vdupq_n_s64(1023)), 52);
    return vmulq_f64(poly, vreinterpretq_f64_s64(exp_bits));
}

static inline float64x2_t b4a_log2(float64x2_t x) {
    const float64x2_t ln2_hi = vdupq_n_f64(0x1.62e42fee00000p-1);
    const float64x2_t ln2_lo = vdupq_n_f64(0x1.a39ef35793c76p-33);
    const uint64x2_t off = vdupq_n_u64(0x3fe6900900000000ULL);
    const uint64x2_t mant_mask = vdupq_n_u64(0x000fffffffffffffULL);

    uint64x2_t ux = vreinterpretq_u64_f64(x);
    uint64x2_t tmp = vsubq_u64(ux, off);
    int64x2_t ki = vshrq_n_s64(vreinterpretq_s64_u64(tmp), 52);
    float64x2_t k = vcvtq_f64_s64(ki);

    uint64x2_t idx_v = vandq_u64(vshrq_n_u64(tmp, 45), vdupq_n_u64(127));
    uint64_t i0 = vgetq_lane_u64(idx_v, 0);
    uint64_t i1 = vgetq_lane_u64(idx_v, 1);

    float64x2_t m = vreinterpretq_f64_u64(vaddq_u64(vandq_u64(tmp, mant_mask), off));

    const b4a_log_entry_t *e0 = &b4a_log_tab[i0];
    const b4a_log_entry_t *e1 = &b4a_log_tab[i1];
    float64x2_t invc = { e0->invc, e1->invc };
    float64x2_t lhi  = { e0->hi,   e1->hi   };
    float64x2_t llo  = { e0->lo,   e1->lo   };

    float64x2_t r = vfmaq_f64(vdupq_n_f64(-1.0), m, invc);
    float64x2_t r2 = vmulq_f64(r, r);

    float64x2_t p23 = vfmaq_f64(vdupq_n_f64(-0.5), r, vdupq_n_f64(1.0 / 3.0));
    float64x2_t p45 = vfmaq_f64(vdupq_n_f64(-0.25), r, vdupq_n_f64(0.2));
    float64x2_t poly = vfmaq_f64(p23, r2, vfmaq_f64(p45, r2, vdupq_n_f64(-1.0 / 6.0)));

    float64x2_t hi = vfmaq_f64(lhi, k, ln2_hi);
    float64x2_t lo = vfmaq_f64(llo, k, ln2_lo);
    lo = vfmaq_f64(lo, r2, poly);
    return vaddq_f64(hi, vaddq_f64(r, lo));
}

#if B4A_W == 2
typedef float64x2_t vad;
typedef uint64x2_t  vam;
static inline vad va_set1(double x)             { return vdupq_n_f64(x); }
static inline vad va_load(const double *p)      { return vld1q_f64(p); }
static inline void va_store(double *p, vad v)   { vst1q_f64(p, v); }
static inline vad va_abs(vad x)                 { return vabsq_f64(x); }
static inline vad va_sqrt(vad x)                { return vsqrtq_f64(x); }
static inline vad va_min(vad a, vad b)          { return vminnmq_f64(a, b); }
static inline vad va_max(vad a, vad b)          { return vmaxnmq_f64(a, b); }
static inline vam va_gt(vad a, vad b)           { return vcgtq_f64(a, b); }
static inline vam va_ge(vad a, vad b)           { return vcgeq_f64(a, b); }
static inline vam va_lt(vad a, vad b)           { return vcltq_f64(a, b); }
static inline vam va_le(vad a, vad b)           { return vcleq_f64(a, b); }
static inline vam va_eq(vad a, vad b)           { return vceqq_f64(a, b); }
static inline vad va_sel(vam m, vad t, vad f)   { return vbslq_f64(m, t, f); }
static inline vad va_exp(vad x)                 { return b4a_exp2(x); }
static inline vad va_log(vad x)                 { return b4a_log2(x); }
static inline vam va_mask_zero(void)            { return vdupq_n_u64(0ULL); }
static inline vam va_mask_all(void)             { return vdupq_n_u64(~0ULL); }
static inline int va_any(vam m)                 { return vmaxvq_u32(vreinterpretq_u32_u64(m)) != 0; }
static inline int va_all(vam m)                 { return vminvq_u32(vreinterpretq_u32_u64(m)) != 0; }
static inline int va_mask_bit(vam m, int lane)  { uint64_t a[2]; vst1q_u64(a, m); return a[lane] != 0; }
#elif B4A_W == 4
typedef double   vad __attribute__((ext_vector_type(4)));
typedef uint64_t vam __attribute__((ext_vector_type(4)));
typedef union { vad v; float64x2_t q[2]; double d[4]; } b4a_u4d;
typedef union { vam v; uint64x2_t  q[2]; uint64_t u[4]; } b4a_u4m;

static inline vad va_set1(double x) {
    b4a_u4d r; float64x2_t q = vdupq_n_f64(x); r.q[0] = q; r.q[1] = q; return r.v;
}
static inline vad va_load(const double *p) {
    b4a_u4d r; r.q[0] = vld1q_f64(p); r.q[1] = vld1q_f64(p + 2); return r.v;
}
static inline void va_store(double *p, vad v) {
    b4a_u4d u; u.v = v; vst1q_f64(p, u.q[0]); vst1q_f64(p + 2, u.q[1]);
}
static inline vad va_abs(vad x) {
    b4a_u4d u, r; u.v = x; r.q[0] = vabsq_f64(u.q[0]); r.q[1] = vabsq_f64(u.q[1]); return r.v;
}
static inline vad va_sqrt(vad x) {
    b4a_u4d u, r; u.v = x; r.q[0] = vsqrtq_f64(u.q[0]); r.q[1] = vsqrtq_f64(u.q[1]); return r.v;
}
static inline vad va_min(vad a, vad b) {
    b4a_u4d ua, ub, r; ua.v = a; ub.v = b;
    r.q[0] = vminnmq_f64(ua.q[0], ub.q[0]); r.q[1] = vminnmq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vad va_max(vad a, vad b) {
    b4a_u4d ua, ub, r; ua.v = a; ub.v = b;
    r.q[0] = vmaxnmq_f64(ua.q[0], ub.q[0]); r.q[1] = vmaxnmq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vam va_gt(vad a, vad b) {
    b4a_u4d ua, ub; b4a_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vcgtq_f64(ua.q[0], ub.q[0]); r.q[1] = vcgtq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vam va_ge(vad a, vad b) {
    b4a_u4d ua, ub; b4a_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vcgeq_f64(ua.q[0], ub.q[0]); r.q[1] = vcgeq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vam va_lt(vad a, vad b) {
    b4a_u4d ua, ub; b4a_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vcltq_f64(ua.q[0], ub.q[0]); r.q[1] = vcltq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vam va_le(vad a, vad b) {
    b4a_u4d ua, ub; b4a_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vcleq_f64(ua.q[0], ub.q[0]); r.q[1] = vcleq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vam va_eq(vad a, vad b) {
    b4a_u4d ua, ub; b4a_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vceqq_f64(ua.q[0], ub.q[0]); r.q[1] = vceqq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vad va_sel(vam m, vad t, vad f) {
    b4a_u4m um; b4a_u4d ut, uf, r; um.v = m; ut.v = t; uf.v = f;
    r.q[0] = vbslq_f64(um.q[0], ut.q[0], uf.q[0]);
    r.q[1] = vbslq_f64(um.q[1], ut.q[1], uf.q[1]);
    return r.v;
}
static inline vad va_exp(vad x) {
    b4a_u4d u, r; u.v = x; r.q[0] = b4a_exp2(u.q[0]); r.q[1] = b4a_exp2(u.q[1]); return r.v;
}
static inline vad va_log(vad x) {
    b4a_u4d u, r; u.v = x; r.q[0] = b4a_log2(u.q[0]); r.q[1] = b4a_log2(u.q[1]); return r.v;
}
static inline vam va_mask_zero(void) {
    b4a_u4m r; uint64x2_t z = vdupq_n_u64(0ULL); r.q[0] = z; r.q[1] = z; return r.v;
}
static inline vam va_mask_all(void) {
    b4a_u4m r; uint64x2_t a = vdupq_n_u64(~0ULL); r.q[0] = a; r.q[1] = a; return r.v;
}
static inline int va_any(vam m) {
    b4a_u4m um; um.v = m;
    return vmaxvq_u32(vreinterpretq_u32_u64(vorrq_u64(um.q[0], um.q[1]))) != 0;
}
static inline int va_all(vam m) {
    b4a_u4m um; um.v = m;
    return vminvq_u32(vreinterpretq_u32_u64(vandq_u64(um.q[0], um.q[1]))) != 0;
}
static inline int va_mask_bit(vam m, int lane) {
    b4a_u4m um; um.v = m; return um.u[lane] != 0;
}
#else
#error "B4A_W must be 2 or 4"
#endif

static inline vad va_log_pos(vad x) { return va_log(x); }
static inline vad va_exp_clamped(vad x, double lo, double hi) {
    return va_exp(va_min(va_max(x, va_set1(lo)), va_set1(hi)));
}

/* Branchless vector limiting helpers (zero branch mispredictions on active misses!) */
static inline vad
va_fetlim(vad vnew, vad vold, vad vto)
{
    const vad z0 = va_set1(0.0);
    const vad two = va_set1(2.0);
    vad vtsthi = va_abs(two * (vold - vto)) + two;
    vad vtstlo = vtsthi * va_set1(0.5) + two;
    vad vtox   = vto + va_set1(3.5);
    vad delv   = vnew - vold;

    vam m_ge_vto  = va_ge(vold, vto);
    vam m_ge_vtox = va_ge(vold, vtox);
    vam m_del_le0 = va_le(delv, z0);

    vad r1_a = va_sel(va_gt(-delv, vtstlo), vold - vtstlo, vnew);
    vad r1_b = va_max(vnew, vto + two);
    vad r1_le = va_sel(va_ge(vnew, vtox), r1_a, r1_b);
    vad r1_gt = va_sel(va_ge(delv, vtstlo), vold + vtstlo, vnew);
    vad r1    = va_sel(m_del_le0, r1_le, r1_gt);

    vad r2_le = va_max(vnew, vto - va_set1(0.5));
    vad r2_gt = va_min(vnew, vto + va_set1(4.0));
    vad r2    = va_sel(m_del_le0, r2_le, r2_gt);

    vad r_ge  = va_sel(m_ge_vtox, r1, r2);

    vad r3_le = va_sel(va_gt(-delv, vtsthi), vold - vtsthi, vnew);
    vad r3_gt = va_min(vnew, vto + va_set1(0.5));
    vad r_lt  = va_sel(m_del_le0, r3_le, r3_gt);

    return va_sel(m_ge_vto, r_ge, r_lt);
}

static inline vad
va_limvds(vad vnew, vad vold)
{
    const vad v35 = va_set1(3.5);
    vam m_ge35 = va_ge(vold, v35);
    vam m_gt   = va_gt(vnew, vold);

    vad r1_gt = va_min(vnew, va_set1(3.0) * vold + va_set1(2.0));
    vad r1_le = va_sel(va_lt(vnew, v35), va_max(vnew, va_set1(2.0)), vnew);
    vad r1    = va_sel(m_gt, r1_gt, r1_le);

    vad r2_gt = va_min(vnew, va_set1(4.0));
    vad r2_le = va_max(vnew, va_set1(-0.5));
    vad r2    = va_sel(m_gt, r2_gt, r2_le);

    return va_sel(m_ge35, r1, r2);
}

static inline double
b4a_fetlim_scalar(double vnew, double vold, double vto)
{
    double vtsthi = fabs(2.0 * (vold - vto)) + 2.0;
    double vtstlo = vtsthi * 0.5 + 2.0;
    double vtox = vto + 3.5;
    double delv = vnew - vold;
    if (vold >= vto) {
        if (vold >= vtox) {
            if (delv <= 0.0) {
                if (vnew >= vtox) {
                    if (-delv > vtstlo) vnew = vold - vtstlo;
                } else {
                    vnew = (vnew > vto + 2.0) ? vnew : (vto + 2.0);
                }
            } else {
                if (delv >= vtstlo) vnew = vold + vtstlo;
            }
        } else {
            if (delv <= 0.0) vnew = (vnew > vto - 0.5) ? vnew : (vto - 0.5);
            else vnew = (vnew < vto + 4.0) ? vnew : (vto + 4.0);
        }
    } else {
        if (delv <= 0.0) {
            if (-delv > vtsthi) vnew = vold - vtsthi;
        } else {
            double vtemp = vto + 0.5;
            if (vnew > vtemp) vnew = vtemp;
        }
    }
    return vnew;
}

static inline double
b4a_limvds_scalar(double vnew, double vold)
{
    if (vold >= 3.5) {
        if (vnew > vold) {
            double v1 = 3.0 * vold + 2.0;
            if (vnew > v1) vnew = v1;
        } else if (vnew < 3.5) {
            vnew = (vnew > 2.0) ? vnew : 2.0;
        }
    } else {
        if (vnew > vold) {
            if (vnew > 4.0) vnew = 4.0;
        } else {
            if (vnew < -0.5) vnew = -0.5;
        }
    }
    return vnew;
}

static inline double
b4a_pnjlim_scalar(double vnew, double vold, double vt, double vcrit, int *icheck)
{
    if (vnew > vcrit && fabs(vnew - vold) > (vt + vt)) {
        if (vold > 0.0) {
            double arg = (vnew - vold) / vt;
            if (arg > 0.0) vnew = vold + vt * (2.0 + log(arg - 2.0));
            else           vnew = vold - vt * (2.0 + log(2.0 - arg));
        } else {
            vnew = vt * log(vnew / vt);
        }
        *icheck = 1;
    } else if (vnew < 0.0) {
        double arg = (vold > 0.0) ? (-vold - 1.0) : (2.0 * vold - 1.0);
        if (vnew < arg) {
            vnew = arg;
            *icheck = 1;
        }
    }
    return vnew;
}

/* Reverse-bias trap-assisted tunneling term (exact match to DEXP) */
static inline void
va_trap_tunnel_term(vam mask, vad vj, vad vt, vad invNvtm, vad revSat,
                    vad *g_io, vad *c_io)
{
    const vad one = va_set1(1.0);
    const vad exp_th = va_set1(B4V_EXP_THRESHOLD);
    const vad nexp_th = va_set1(-B4V_EXP_THRESHOLD);
    const vad max_exp = va_set1(B4V_MAX_EXP);
    const vad min_exp = va_set1(B4V_MIN_EXP);

    vad diff = vt - vj;
    vam m_small = va_lt(diff, vt * va_set1(1.0e-3));
    vad T9_reg = one / va_sel(m_small, one, diff);
    vad T0_sm  = -vj * invNvtm * va_set1(1.0e3);
    vad T0_reg = -vj * invNvtm * vt * T9_reg;
    vad T0     = va_sel(m_small, T0_sm, T0_reg);

    vam m_hi  = va_gt(T0, exp_th);
    vam m_lo  = va_lt(T0, nexp_th);
    vam m_mid = (~m_hi) & (~m_lo) & mask;

    vad B = va_sel(m_hi, max_exp * (one + T0 - exp_th), min_exp);
    vad C = va_sel(m_hi, max_exp, va_set1(0.0));
    if (va_any(m_mid)) {
        vad ex = va_exp_clamped(T0, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
        B = va_sel(m_mid, ex, B);
        C = va_sel(m_mid, ex, C);
    }

    vad dT0_sm  = invNvtm * va_set1(1.0e3);
    vad dT0_reg = vt * invNvtm * (T9_reg + vj * T9_reg * T9_reg);
    vad dB_dVb  = C * va_sel(m_small, dT0_sm, dT0_reg);

    *g_io = va_sel(mask, *g_io + revSat * dB_dVb, *g_io);
    *c_io = va_sel(mask, *c_io - revSat * (B - one), *c_io);
}

/* Poly gate depletion (exact match to BSIM4v5/v6 polyDepletion with mtrlMod=0) */
static inline void
va_poly_depletion(vad phi_vfb, vad polyT1, vam m_ngate_valid,
                  vad Vgs, vad *Vgs_eff, vad *dVgs_eff_dVg)
{
    vam m = m_ngate_valid & va_gt(Vgs, phi_vfb);
    if (!va_any(m)) {
        *Vgs_eff = Vgs;
        *dVgs_eff_dVg = va_set1(1.0);
        return;
    }
    vad one = va_set1(1.0);
    vad T8 = va_max(Vgs - phi_vfb, va_set1(0.0));
    vad invT1 = one / polyT1;
    vad T4 = va_sqrt(one + va_set1(2.0) * T8 * invT1);
    vad T2 = va_set1(2.0) * T8 / (T4 + one);
    vad T3 = va_set1(0.5) * T2 * T2 * invT1;
    vad T7 = va_set1(1.07) - T3;
    vad T6 = va_sqrt(T7 * T7 + va_set1(0.224));
    vad T5 = va_set1(1.12) - va_set1(0.5) * (T7 + T6);
    vad eff = Vgs - T5;
    vad deff = one - (va_set1(0.5) - va_set1(0.5) / T4) * (one + T7 / T6);
    *Vgs_eff = va_sel(m, eff, Vgs);
    *dVgs_eff_dVg = va_sel(m, deff, one);
}

/* ========================================================================= */
/* Apple Silicon Cache-Line-Engineered Data Structures                       */
/* ========================================================================= */

typedef struct __attribute__((aligned(128))) {
    double type;
    double nf;
    double vth0_signed;
    double vfb_plus_phi;
    double vfbzb;
    double vbsc;
    double phi;
    double phi_095;
    double sqrtPhi;
    double V0;
    double vtm;
    double coxe;
    double invCoxe;
    double coxp;
    double toxe;
    double invToxe;
    double toxp;
    double tox_ctm;
    double inv_tox_ctm;
    double inv_2e8_toxp;
    double link_ctm;
    double leff;
    double invLeff;
    double sqrtLeff;
    double weff;
    double weffCV;
    double leffCV;
    double CoxWL;
    double CoxWL_ov_coxe;
    double ados1p9;
    double bdos0p7;
    double invNvtms;
    double invNvtmd;
    double invNvtmrs_s;
    double invNvtmrs_d;
    double invNvtmrssw_s;
    double invNvtmrssw_d;
    double invNvtmrsswg_s;
    double invNvtmrsswg_d;
    double vtss;
    double vtsd;
    double vtssws;
    double vtsswd;
    double vtsswgs;
    double vtsswgd;
    double xdep0OverSqrtPhi;
    double factor1;
    double dvt0;
    double dvt1_leff;
    double dvt2;
    double dvt0w;
    double dvt1w_weff_leff;
    double dvt2w;
    double k1;
    double k1ox;
    double k2ox;
    double k3;
    double k3b;
    double lpebFactor;
    double vthNarrowW;
    double vth_const_T1;
    double kt2_TempRatio;
    double eta0;
    double etab;
    double theta0vb0;
    double nfactor;
    double cit;
    double cdsc;
    double cdscb;
    double cdscd;
    double dvtp0;
    double dvtp1;
    double polyT1;
    double mstar;
    double voffcbn;
    double coxe_ov_cdep0;
    double dwg;
    double dwb;
    double prwg;
    double prwb;
    double rds0_half;
    double rdswmin;
    double xj;
    double a0;
    double b0_ov_wb1;
    double ags_a0;
    double keta;
    double abulkCVfactor;
    double ua;
    double ub;
    double uc;
    double ud;
    double u0temp;
    double vsattemp;
    double a1;
    double a2;
    double delta;
    double lambda;
    double litl;
    double fprout;
    double pvag;
    double pclm;
    double inv_pclm_litl;
    double thetaRout;
    double inv_thetaRout;
    double pdiblb;
    double pdits;
    double inv_pdits;
    double pditsd;
    double pditsl_term;
    double pscbe1_litl;
    double inv_pscbe2;
    double alpha_ov_leff;
    double beta0;
    double inv_vtl_tfactor;
    double vtfbphi2;
    double inv_3toxe;
    double agidl_weffCJ;
    double bgidl;
    double cgidl;
    double egidl;
    double agisl_weffCJ;
    double bgisl;
    double cgisl;
    double egisl;
    double noff;
    double voffcv;
    double acde;
    double ldeb;
    double moin;
    double cgdo;
    double cgso;
    double cgbo;
    double weffCV_cgdl;
    double ckappad;
    double inv_ckappad;
    double weffCV_cgsl;
    double ckappas;
    double inv_ckappas;
    double PhiBS;
    double invPhiBS;
    double PhiBSWS;
    double invPhiBSWS;
    double PhiBSWGS;
    double invPhiBSWGS;
    double PhiBD;
    double invPhiBD;
    double PhiBSWD;
    double invPhiBSWD;
    double PhiBSWGD;
    double invPhiBSWGD;
    double MJS;
    double inv1mMJS;
    double MJSWS;
    double inv1mMJSWS;
    double MJSWGS;
    double inv1mMJSWGS;
    double MJD;
    double inv1mMJD;
    double MJSWD;
    double inv1mMJSWD;
    double MJSWGD;
    double inv1mMJSWGD;
    double vcrit;
    uint8_t f_ngate_valid;
    uint8_t f_dvt0w;
    uint8_t f_dvtp0;
    uint8_t f_fprout;
    uint8_t f_pclm;
    uint8_t f_thetaRout;
    uint8_t f_pdits;
    uint8_t f_pscbe;
    uint8_t f_isub;
    uint8_t f_vtl;
    uint8_t f_lambda;
    uint8_t f_gidl_en;
    uint8_t f_gisl_en;
    uint8_t f_xpart_gt05;
    uint8_t f_xpart_lt05;
    uint8_t f_mjs_half;
    uint8_t f_mjsws_half;
    uint8_t f_mjswgs_half;
    uint8_t f_mjd_half;
    uint8_t f_mjswd_half;
    uint8_t f_mjswgd_half;
    uint8_t f_k1ox_zero;
    uint8_t f_k1ox_le0;
    uint8_t f_vfbzb_le0;
    uint8_t f_nf_ne1;
    uint8_t f_has_a1;
    uint8_t is_v6;
    uint8_t has_ud;
    uint8_t xpart_mode;
    uint8_t has_trap_tunnel;
} BSIM4ApplePParam;

typedef struct __attribute__((aligned(64))) {
    double SourceSatCurrent;
    double DrainSatCurrent;
    double vjsmFwd;
    double vjdmFwd;
    double IVjsmFwd;
    double IVjdmFwd;
    double czbs;
    double czbssw;
    double czbsswg;
    double czbd;
    double czbdsw;
    double czbdswg;
    double jctS_sum;
    double jctD_sum;
    double jctS_lin;
    double jctD_lin;
    double SjctRevSat;
    double DjctRevSat;
    double SswRevSat;
    double DswRevSat;
    double SswgRevSat;
    double DswgRevSat;
    uint16_t flags;
} BSIM4AppleJct;

/* EXACTLY 128 BYTES = 1 Apple Silicon cache line!
 * L1 hits with pack-level step validation + delta-stamping touch ONLY this 1 cache line! */
typedef struct __attribute__((aligned(128))) {
    int dNodePrime, gNodePrime, sNodePrime, bNodePrime;  /* 16B */
    int stateBase, instIdx;                              /* 24B */
    uint8_t has_dint, has_sint, phys_valid, stamp_valid; /* 28B */
    int pad0;                                            /* 32B */
    double type, last_vds, last_vgs, last_vbs;           /* 64B */
    double Idtot, cd_g, cd_d, cd_b;                      /* 96B */
    double Ibtot, cb_g, cb_d, cb_b;                      /* 128B */
} BSIM4AppleCheck;

/* Compact per-instance stamp + limiter metadata (NOT touched on L1 delta-stamp hits!) */
typedef struct __attribute__((aligned(64))) {
    double st[20];
    int    kluOff[16];
    double m, mgd, mgs, vcrit, von;
    double drainCond, sourceCond;
} BSIM4AppleStamp;

typedef struct __attribute__((aligned(64))) {
    double m, mtype;
    double mdc44, mdc46, mdc47, mdc48;
    double mdc51, mdc52, mdc53, mdc55;
    double mdc58, mdc59, mdc60, mdc61;
    double mdc_rhsd, mdc_rhsb, mdc_rhss;
    double Cggb, Cgdb, Cgsb;
    double Cdgb, Cddb, Cdsb;
    double Csgb, Csdb, Cssb;
    double Cbgb, Cbdb, Cbsb;
    double q_lin_g, q_lin_d, q_lin_b;
    double qg_st, qd_st, qb_st;
    double base_cqg, base_cqd, base_cqb;
    double a_vds, a_vgs, a_vbs, a_vbd;
} BSIM4AppleL2;

typedef struct {
    double von[B4A_W];
    int    mode[B4A_W];
    double cd[B4A_W], gds[B4A_W], gm[B4A_W], gmbs[B4A_W];
    double gbd[B4A_W], gbs[B4A_W], cbs[B4A_W], cbd[B4A_W];
    double csub[B4A_W], gbds[B4A_W], gbgs[B4A_W], gbbs[B4A_W];
    double Igidl[B4A_W], ggidld[B4A_W], ggidlg[B4A_W], ggidlb[B4A_W];
    double Igisl[B4A_W], ggisls[B4A_W], ggislg[B4A_W], ggislb[B4A_W];
} BSIM4AppleOut;

typedef struct {
    BSIM4ApplePParam pp;
    int offset;
    int count;
} BSIM4AppleGroup;

typedef struct {
    BSIM4AppleGroup *groups;
    BSIM4AppleCheck *checks;
    BSIM4AppleStamp *stamps;
    BSIM4AppleL2    *l2s;
    BSIM4AppleJct   *jcts;
    int             *eval_q;
    double          *eval_vds;
    double          *eval_vgs;
    double          *eval_vbs;
    int              nGroups;
    int              nElig;
    /* Pack-level step cache metadata (checked ONCE per CKTload instead of per transistor!) */
    double           last_time, last_ag0, last_ag1, last_ag2, last_gmin;
    const double    *last_s0, *last_s1, *last_s2;
    int              last_int_mode;
    int              step_valid;
} BSIM4ApplePack;

typedef struct __attribute__((aligned(64))) {
    vad vds;
    vad vgs;
    vad vbs;
    vad vbd;
    vad vgd;
    vad vgb;
    vad Vds;
    vad Vgs_eff;
    vad dVgs_eff_dVg;
    vad vgs_eff;
    vad dvgs_eff_dvg;
    vad vgd_eff;
    vad dvgd_eff_dvg;
    vad Vbseff;
    vad dVbseff_dVb;
    vad Phis;
    vad Vth;
    vad dVth_dVd;
    vad dVth_dVb;
    vad n;
    vad dn_dVd;
    vad dn_dVb;
    vad Vgst;
    vad Vgsteff;
    vad dVgsteff_dVg;
    vad dVgsteff_dVd;
    vad dVgsteff_dVb;
    vad Weff;
    vad dWeff_dVg;
    vad dWeff_dVb;
    vad Rds;
    vad dRds_dVg;
    vad dRds_dVb;
    vad Abulk;
    vad dAbulk_dVg;
    vad dAbulk_dVb;
    vad Abulk0;
    vad dAbulk0_dVb;
    vad ueff;
    vad dueff_dVg;
    vad dueff_dVd;
    vad dueff_dVb;
    vad Esat;
    vad EsatL;
    vad dEsatL_dVg;
    vad dEsatL_dVd;
    vad dEsatL_dVb;
    vad Vdsat;
    vad dVdsat_dVg;
    vad dVdsat_dVd;
    vad dVdsat_dVb;
    vad Vdseff;
    vad dVdseff_dVg;
    vad dVdseff_dVd;
    vad dVdseff_dVb;
    vad WVCoxRds;
    vad Lambda;
    vad diffVds;
    vad Vgst2Vtm;
    vad tmp1_lam;
    vad tmp2_rw;
    vad tmp3_rw;
    vam m_fwd;
} BSIM4ApplePipe;

#if B4A_W == 2
#define va_jct_gather(jpts, field) ((vad){ (jpts)[0]->field, (jpts)[1]->field })
static inline vam va_jct_mask(const BSIM4AppleJct * const jpts[2], uint16_t bit) {
    vam r = { (jpts[0]->flags & bit) ? ~0ULL : 0ULL, (jpts[1]->flags & bit) ? ~0ULL : 0ULL };
    return r;
}
#else
#define va_jct_gather(jpts, field) ((vad){ (jpts)[0]->field, (jpts)[1]->field, (jpts)[2]->field, (jpts)[3]->field })
static inline vam va_jct_mask(const BSIM4AppleJct * const jpts[4], uint16_t bit) {
    b4a_u4m r;
    r.u[0] = (jpts[0]->flags & bit) ? ~0ULL : 0ULL;
    r.u[1] = (jpts[1]->flags & bit) ? ~0ULL : 0ULL;
    r.u[2] = (jpts[2]->flags & bit) ? ~0ULL : 0ULL;
    r.u[3] = (jpts[3]->flags & bit) ? ~0ULL : 0ULL;
    return r.v;
}
#endif

/* ========================================================================= */
/* Inline Scalar Level-1 Sync & Level-2 Integrate/Assemble (W=1)             */
/* ========================================================================= */

static inline void
bsim4_apple_l1_sync_one(
    const BSIM4AppleCheck * restrict c,
    const BSIM4AppleL2 * restrict l2,
    double vds, double vgs, double vbs,
    double ag0,
    double * restrict s0)
{
    double dvds = vds - c->last_vds;
    double dvgs = vgs - c->last_vgs;
    double dvbs = vbs - c->last_vbs;
    double dvgb = dvgs - dvbs;
    double dvbd = dvbs - dvds;
    double dqg = l2->Cggb * dvgb - l2->Cgdb * dvbd - l2->Cgsb * dvbs;
    double dqd = l2->Cdgb * dvgb - l2->Cddb * dvbd - l2->Cdsb * dvbs;
    double dqb = l2->Cbgb * dvgb - l2->Cbdb * dvbd - l2->Cbsb * dvbs;
    int st = c->stateBase;
    s0[st + 0]  = vbs - vds;
    s0[st + 1]  = vbs;
    s0[st + 2]  = vgs;
    s0[st + 3]  = vds;
    s0[st + 11] = l2->qb_st + dqb;
    s0[st + 12] = l2->base_cqb + ag0 * dqb;
    s0[st + 13] = l2->qg_st + dqg;
    s0[st + 14] = l2->base_cqg + ag0 * dqg;
    s0[st + 15] = l2->qd_st + dqd;
    s0[st + 16] = l2->base_cqd + ag0 * dqd;
}

static inline void
bsim4_apple_l2_assemble_one(
    const BSIM4AppleCheck * restrict c,
    BSIM4AppleL2 * restrict l2,
    BSIM4AppleStamp * restrict sp,
    double vds, double vgs, double vbs,
    int has_dv,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2)
{
    double dqg = 0.0, dqd = 0.0, dqb = 0.0;
    double qb_st = l2->qb_st, qg_st = l2->qg_st, qd_st = l2->qd_st;
    double qb, qg, qd;
    int st = c->stateBase;
    if (has_dv) {
        double dvds = vds - c->last_vds;
        double dvgs = vgs - c->last_vgs;
        double dvbs = vbs - c->last_vbs;
        double dvgb = dvgs - dvbs;
        double dvbd = dvbs - dvds;
        dqg = l2->Cggb * dvgb - l2->Cgdb * dvbd - l2->Cgsb * dvbs;
        dqd = l2->Cdgb * dvgb - l2->Cddb * dvbd - l2->Cdsb * dvbs;
        dqb = l2->Cbgb * dvgb - l2->Cbdb * dvbd - l2->Cbsb * dvbs;
        qb = qb_st + dqb;
        qg = qg_st + dqg;
        qd = qd_st + dqd;
        s0[st + 0] = vbs - vds;
        s0[st + 1] = vbs;
        s0[st + 2] = vgs;
        s0[st + 3] = vds;
    } else {
        qb = qb_st;
        qg = qg_st;
        qd = qd_st;
        s0[st + 0] = l2->a_vbd;
        s0[st + 1] = l2->a_vbs;
        s0[st + 2] = l2->a_vgs;
        s0[st + 3] = l2->a_vds;
    }
    s0[st + 11] = qb;
    s0[st + 13] = qg;
    s0[st + 15] = qd;

    double base_cqb, base_cqg, base_cqd;
    if (is_gear2) {
        base_cqb = ag2 * s2[st + 11] + ag1 * s1[st + 11] + ag0 * qb_st;
        base_cqg = ag2 * s2[st + 13] + ag1 * s1[st + 13] + ag0 * qg_st;
        base_cqd = ag2 * s2[st + 15] + ag1 * s1[st + 15] + ag0 * qd_st;
    } else if (is_trap2) {
        base_cqb = -s1[st + 12] * ag1 + ag0 * (qb_st - s1[st + 11]);
        base_cqg = -s1[st + 14] * ag1 + ag0 * (qg_st - s1[st + 13]);
        base_cqd = -s1[st + 16] * ag1 + ag0 * (qd_st - s1[st + 15]);
    } else {
        base_cqb = ag0 * qb_st + ag1 * s1[st + 11];
        base_cqg = ag0 * qg_st + ag1 * s1[st + 13];
        base_cqd = ag0 * qd_st + ag1 * s1[st + 15];
    }
    l2->base_cqb = base_cqb;
    l2->base_cqg = base_cqg;
    l2->base_cqd = base_cqd;
    double cqb = has_dv ? (base_cqb + ag0 * dqb) : base_cqb;
    double cqg = has_dv ? (base_cqg + ag0 * dqg) : base_cqg;
    double cqd = has_dv ? (base_cqd + ag0 * dqd) : base_cqd;
    s0[st + 12] = cqb;
    s0[st + 14] = cqg;
    s0[st + 16] = cqd;

    double vm_ag0 = l2->m * ag0;
    double mtype = l2->mtype;
    double mceqqg = (cqg + ag0 * (l2->q_lin_g - dqg)) * mtype;
    double mceqqd = (cqd + ag0 * (l2->q_lin_d - dqd)) * mtype;
    double mceqqb = (cqb + ag0 * (l2->q_lin_b - dqb)) * mtype;

    double * restrict dst = sp->st;
    dst[0] = l2->mdc_rhsd - mceqqd;
    dst[1] = mceqqg;
    dst[2] = l2->mdc_rhsb - mceqqb;
    dst[3] = l2->mdc_rhss + mceqqg + mceqqb + mceqqd;

    double v34 = l2->Cggb * vm_ag0;
    double v35 = l2->Cgdb * vm_ag0;
    double v36 = l2->Cgsb * vm_ag0;
    dst[4] = v34;
    dst[5] = v35;
    dst[6] = v36;
    dst[7] = -(v34 + v35 + v36);

    double mgcdgb = l2->Cdgb * vm_ag0;
    double mgcddb = l2->Cddb * vm_ag0;
    double mgcdsb = l2->Cdsb * vm_ag0;
    dst[8]  = l2->mdc44 + mgcddb;
    dst[9]  = l2->mdc46 + mgcdgb;
    dst[10] = l2->mdc47 - mgcdsb;
    dst[11] = l2->mdc48 + (mgcdgb + mgcddb + mgcdsb);

    double mgcsgb = l2->Csgb * vm_ag0;
    double mgcsdb = l2->Csdb * vm_ag0;
    double mgcssb = l2->Cssb * vm_ag0;
    dst[12] = l2->mdc51 - mgcsdb;
    dst[13] = l2->mdc52 + mgcsgb;
    dst[14] = l2->mdc53 + mgcssb;
    dst[15] = l2->mdc55 + (mgcsgb + mgcsdb + mgcssb);

    double mgcbgb = l2->Cbgb * vm_ag0;
    double mgcbdb = l2->Cbdb * vm_ag0;
    double mgcbsb = l2->Cbsb * vm_ag0;
    dst[16] = l2->mdc58 + mgcbdb;
    dst[17] = l2->mdc59 + mgcbgb;
    dst[18] = l2->mdc60 + mgcbsb;
    dst[19] = l2->mdc61 - (mgcbgb + mgcbdb + mgcbsb);
}

/* Single shared compiled kernel entry point (defined ONCE in b4v6ld.c, linked by b4v5ld.c) */
extern void bsim4_apple_eval_batch(
    const BSIM4ApplePParam * restrict pp,
    const BSIM4AppleJct * const jpts[B4A_W],
    BSIM4AppleCheck * const cpts[B4A_W],
    BSIM4AppleStamp * const spts[B4A_W],
    BSIM4AppleL2 * const l2pts[B4A_W],
    int nact,
    const double vds_in[B4A_W],
    const double vgs_in[B4A_W],
    const double vbs_in[B4A_W],
    double ckt_gmin,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2,
    BSIM4AppleOut * restrict out,
    int need_aos);

#ifdef B4_APPLE_DEFINE_KERNEL

#define B4A_SCATTER(ptrs, field, vec) do { \
    double _tmp[B4A_W]; \
    va_store(_tmp, (vec)); \
    for (int _l = 0; _l < nact; _l++) (ptrs)[_l]->field = _tmp[_l]; \
} while (0)

__attribute__((noinline))
static void
bsim4_apple_p1_vth_mob(
    const BSIM4ApplePParam * restrict pp,
    BSIM4ApplePipe * restrict p,
    vad vds, vad vgs, vad vbs)
{
    const vad z0 = va_set1(0.0);
    const vad one = va_set1(1.0);
    const vad half = va_set1(0.5);
    const vad two = va_set1(2.0);
    const vad max_exp = va_set1(B4V_MAX_EXP);
    const vad min_exp = va_set1(B4V_MIN_EXP);
    const vad exp_th = va_set1(B4V_EXP_THRESHOLD);
    const vad nexp_th = va_set1(-B4V_EXP_THRESHOLD);
    const vad epssi = va_set1(B4V_EPSSI);
    (void)max_exp; (void)min_exp;

    vad vbd = vbs - vds;
    vad vgd = vgs - vds;
    vad vgb = vgs - vbs;


    /* 2. Forward / Reverse mode swap                                        */
    /* ===================================================================== */
    vam m_fwd = va_ge(vds, z0);
    vad Vds = va_sel(m_fwd, vds, -vds);
    vad Vgs = va_sel(m_fwd, vgs, vgd);
    vad Vbs = va_sel(m_fwd, vbs, vbd);

    /* ===================================================================== */
    /* 3. Vbseff, Phis, Xdep, Vth                                            */
    /* ===================================================================== */
    vad Vbseff, dVbseff_dVb;
    {
        vad T0 = Vbs - va_set1(pp->vbsc) - va_set1(0.001);
        vad T1 = va_sqrt(T0 * T0 - va_set1(0.004) * va_set1(pp->vbsc));
        vam m_ge0 = va_ge(T0, z0);
        vad v_pos = va_set1(pp->vbsc) + half * (T0 + T1);
        vad dv_pos = half * (one + T0 / T1);
        vad T2 = va_set1(-0.002) / (T1 - T0);
        vad v_neg = va_set1(pp->vbsc) * (one + T2);
        vad dv_neg = T2 * va_set1(pp->vbsc) / T1;
        Vbseff = va_sel(m_ge0, v_pos, v_neg);
        dVbseff_dVb = va_sel(m_ge0, dv_pos, dv_neg);

        vad T9 = va_set1(pp->phi_095);
        T0 = T9 - Vbseff - va_set1(0.001);
        T1 = va_sqrt(T0 * T0 + va_set1(0.004) * T9);
        Vbseff = T9 - half * (T0 + T1);
        dVbseff_dVb = dVbseff_dVb * (half * (one + T0 / T1));
    }

    vad Phis = va_set1(pp->phi) - Vbseff;
    vad sqrtPhis = va_sqrt(Phis);
    vad dsqrtPhis_dVb = va_set1(-0.5) / sqrtPhis;
    vad Xdep = va_set1(pp->xdep0OverSqrtPhi) * sqrtPhis;
    vad dXdep_dVb = va_set1(pp->xdep0OverSqrtPhi) * dsqrtPhis_dVb;

    vad sqrtXdep = va_sqrt(Xdep);
    vad invSqrtXdep = one / sqrtXdep;

    /* lt1 */
    vad lt1, dlt1_dVb;
    {
        vad T0 = va_set1(pp->dvt2) * Vbseff;
        vam m_ge = va_ge(T0, va_set1(-0.5));
        vad T4 = one / (va_set1(3.0) + va_set1(8.0) * T0);
        vad T1 = va_sel(m_ge, one + T0, (one + va_set1(3.0) * T0) * T4);
        vad T2 = va_sel(m_ge, va_set1(pp->dvt2), va_set1(pp->dvt2) * T4 * T4);
        lt1 = va_set1(pp->factor1) * sqrtXdep * T1;
        dlt1_dVb = va_set1(pp->factor1) * (half * invSqrtXdep * T1 * dXdep_dVb + sqrtXdep * T2);
    }

    /* Theta0 */
    vad Theta0, dTheta0_dVb;
    {
        vad T0 = va_set1(pp->dvt1_leff) / lt1;
        vam m_lt = va_lt(T0, exp_th);
        vad th_hi = va_set1(1.0 / (B4V_MAX_EXP - 2.0));
        if (va_any(m_lt)) {
            vad T1 = va_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            vad T2 = T1 - one;
            vad T4 = T2 * T2 + two * T1 * min_exp;
            vad invT4 = one / T4;
            vad th = T1 * invT4;
            vad dT1 = -T0 * T1 * dlt1_dVb / lt1;
            vad dth = dT1 * (T4 - two * T1 * (T2 + min_exp)) * (invT4 * invT4);
            Theta0 = va_sel(m_lt, th, th_hi);
            dTheta0_dVb = va_sel(m_lt, dth, z0);
        } else {
            Theta0 = th_hi;
            dTheta0_dVb = z0;
        }
    }
    vad Delt_vth = va_set1(pp->dvt0) * Theta0 * va_set1(pp->V0);
    vad dDelt_vth_dVb = va_set1(pp->dvt0) * dTheta0_dVb * va_set1(pp->V0);

    /* dvt0w narrow-width SCE (skip completely if all lanes have dvt0w == 0) */
    vad T2_dvt0w = z0, dT2_dvt0w_dVb = z0;
    if (pp->f_dvt0w) {
        vad T0w = va_set1(pp->dvt2w) * Vbseff;
        vam m_ge = va_ge(T0w, va_set1(-0.5));
        vad T4w = one / (va_set1(3.0) + va_set1(8.0) * T0w);
        vad T1w = va_sel(m_ge, one + T0w, (one + va_set1(3.0) * T0w) * T4w);
        vad T2w = va_sel(m_ge, va_set1(pp->dvt2w), va_set1(pp->dvt2w) * T4w * T4w);
        vad ltw = va_set1(pp->factor1) * sqrtXdep * T1w;
        vad dltw_dVb = va_set1(pp->factor1) * (half * invSqrtXdep * T1w * dXdep_dVb + sqrtXdep * T2w);
        vad T0 = va_set1(pp->dvt1w_weff_leff) / ltw;
        vam m_lt = va_lt(T0, exp_th) & (pp->f_dvt0w ? va_mask_all() : va_mask_zero());
        vad t5_hi = va_set1(1.0 / (B4V_MAX_EXP - 2.0));
        vad T5 = t5_hi, dT5_dVb = z0;
        if (va_any(m_lt)) {
            vad T1 = va_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            vad T2 = T1 - one;
            vad T4 = T2 * T2 + two * T1 * min_exp;
            vad invT4 = one / T4;
            vad dT1 = -T0 * T1 * dltw_dVb / ltw;
            T5 = va_sel(m_lt, T1 * invT4, t5_hi);
            dT5_dVb = va_sel(m_lt, dT1 * (T4 - two * T1 * (T2 + min_exp)) * (invT4 * invT4), z0);
        }
        T2_dvt0w = va_sel((pp->f_dvt0w ? va_mask_all() : va_mask_zero()), va_set1(pp->dvt0w) * T5 * va_set1(pp->V0), z0);
        dT2_dvt0w_dVb = va_sel((pp->f_dvt0w ? va_mask_all() : va_mask_zero()), va_set1(pp->dvt0w) * dT5_dVb * va_set1(pp->V0), z0);
    }

    /* DIBL shift */
    vad dDIBL_Sft_dVd, DIBL_Sft, T4_dibl;
    {
        vad T3 = va_set1(pp->eta0) + va_set1(pp->etab) * Vbseff;
        vam m_lt = va_lt(T3, va_set1(1.0e-4));
        vad T9 = one / (va_set1(3.0) - va_set1(2.0e4) * T3);
        T3 = va_sel(m_lt, (va_set1(2.0e-4) - T3) * T9, T3);
        T4_dibl = va_sel(m_lt, T9 * T9, one);
        dDIBL_Sft_dVd = T3 * va_set1(pp->theta0vb0);
        DIBL_Sft = dDIBL_Sft_dVd * Vds;
    }

    vad Vth = va_set1(pp->vth_const_T1) + va_set1(pp->k1ox) * sqrtPhis * va_set1(pp->lpebFactor)
            - va_set1(pp->k2ox) * Vbseff - Delt_vth - T2_dvt0w
            + va_set1(pp->k3b) * Vbseff * va_set1(pp->vthNarrowW)
            + va_set1(pp->kt2_TempRatio) * Vbseff - DIBL_Sft;
    vad dVth_dVb = va_set1(pp->lpebFactor) * va_set1(pp->k1ox) * dsqrtPhis_dVb - va_set1(pp->k2ox)
                 - dDelt_vth_dVb - dT2_dvt0w_dVb + va_set1(pp->k3b) * va_set1(pp->vthNarrowW)
                 - va_set1(pp->etab) * Vds * va_set1(pp->theta0vb0) * T4_dibl + va_set1(pp->kt2_TempRatio);
    vad dVth_dVd = -dDIBL_Sft_dVd;

    /* Subthreshold swing n */
    vad n, dn_dVb, dn_dVd;
    {
        vad tmp1 = epssi / Xdep;
        vad tmp2 = va_set1(pp->nfactor) * tmp1;
        vad tmp3 = va_set1(pp->cdsc) + va_set1(pp->cdscb) * Vbseff + va_set1(pp->cdscd) * Vds;
        vad tmp4 = (tmp2 + tmp3 * Theta0 + va_set1(pp->cit)) * va_set1(pp->invCoxe);
        vad base_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb + va_set1(pp->cdscb) * Theta0) * va_set1(pp->invCoxe);
        vad base_dVd = va_set1(pp->cdscd) * Theta0 * va_set1(pp->invCoxe);
        vam m_ge = va_ge(tmp4, va_set1(-0.5));
        vad T0 = one / (va_set1(3.0) + va_set1(8.0) * tmp4);
        vad T0sq = T0 * T0;
        n = va_sel(m_ge, one + tmp4, (one + va_set1(3.0) * tmp4) * T0);
        dn_dVb = va_sel(m_ge, base_dVb, base_dVb * T0sq);
        dn_dVd = va_sel(m_ge, base_dVd, base_dVd * T0sq);
    }

    /* Pocket implant Vth correction (dvtp0 > 0) */
    if (pp->f_dvtp0) {
        vad T0 = -va_set1(pp->dvtp1) * Vds;
        vam m_ge = va_ge(T0, nexp_th) & (pp->f_dvtp0 ? va_mask_all() : va_mask_zero());
        vad T2 = min_exp, dT2_dVd = z0;
        if (va_any(m_ge)) {
            vad ex = va_exp_clamped(T0, -B4V_EXP_THRESHOLD, 100.0);
            T2 = va_sel(m_ge, ex, min_exp);
            dT2_dVd = va_sel(m_ge, -va_set1(pp->dvtp1) * ex, z0);
        }
        vad T3 = va_set1(pp->leff) + va_set1(pp->dvtp0) * (one + T2);
        vad dT3_dVd = va_set1(pp->dvtp0) * dT2_dVd;
        vad T4 = va_set1(pp->vtm) * va_log_pos(va_set1(pp->leff) / T3);
        vad dT4_dVd = -va_set1(pp->vtm) * dT3_dVd / T3;
        vad dDITS_dVd = dn_dVd * T4 + n * dT4_dVd;
        vad dDITS_dVb = T4 * dn_dVb;
        Vth      = va_sel((pp->f_dvtp0 ? va_mask_all() : va_mask_zero()), Vth - n * T4, Vth);
        dVth_dVd = va_sel((pp->f_dvtp0 ? va_mask_all() : va_mask_zero()), dVth_dVd - dDITS_dVd, dVth_dVd);
        dVth_dVb = va_sel((pp->f_dvtp0 ? va_mask_all() : va_mask_zero()), dVth_dVb - dDITS_dVb, dVth_dVb);
    }
    /* Vth saved in pipe->Vth */

    /* ===================================================================== */
    /* 4. Poly Depletion & Vgsteff                                           */
    /* ===================================================================== */
    vad vgs_eff, dvgs_eff_dvg, vgd_eff, dvgd_eff_dvg;
    va_poly_depletion(va_set1(pp->vfb_plus_phi), va_set1(pp->polyT1), (pp->f_ngate_valid ? va_mask_all() : va_mask_zero()), vgs, &vgs_eff, &dvgs_eff_dvg);
    va_poly_depletion(va_set1(pp->vfb_plus_phi), va_set1(pp->polyT1), (pp->f_ngate_valid ? va_mask_all() : va_mask_zero()), vgd, &vgd_eff, &dvgd_eff_dvg);
    vad Vgs_eff = va_sel(m_fwd, vgs_eff, vgd_eff);
    vad dVgs_eff_dVg = va_sel(m_fwd, dvgs_eff_dvg, dvgd_eff_dvg);

    vad Vgst = Vgs_eff - Vth;

    vad Vgsteff, dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb;
    {
        vad T0 = n * va_set1(pp->vtm);
        vad T1 = va_set1(pp->mstar) * Vgst;
        vad T2 = T1 / T0;
        vam m_hi = va_gt(T2, exp_th);
        vam m_lo = va_lt(T2, nexp_th);
        vam m_mid = (~m_hi) & (~m_lo);

        vad T10_hi = T1;
        vad dT10g_hi = va_set1(pp->mstar) * dVgs_eff_dVg;
        vad dT10d_hi = -dVth_dVd * va_set1(pp->mstar);
        vad dT10b_hi = -dVth_dVb * va_set1(pp->mstar);

        vad t10_base = va_set1(pp->vtm) * va_set1(B4V_LOG1P_MIN_EXP);
        vad T10_lo = t10_base * n;
        vad dT10d_lo = t10_base * dn_dVd;
        vad dT10b_lo = t10_base * dn_dVb;

        vad T10 = va_sel(m_hi, T10_hi, T10_lo);
        vad dT10g = va_sel(m_hi, dT10g_hi, z0);
        vad dT10d = va_sel(m_hi, dT10d_hi, dT10d_lo);
        vad dT10b = va_sel(m_hi, dT10b_hi, dT10b_lo);
        if (va_any(m_mid)) {
            vad ex = va_exp_clamped(T2, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            vad T3 = va_set1(pp->vtm) * va_log_pos(one + ex);
            vad dg0 = va_set1(pp->mstar) * ex / (one + ex);
            vad inv_n = one / n;
            T10   = va_sel(m_mid, n * T3, T10);
            dT10b = va_sel(m_mid, T3 * dn_dVb - dg0 * (dVth_dVb + Vgst * dn_dVb * inv_n), dT10b);
            dT10d = va_sel(m_mid, T3 * dn_dVd - dg0 * (dVth_dVd + Vgst * dn_dVd * inv_n), dT10d);
            dT10g = va_sel(m_mid, dg0 * dVgs_eff_dVg, dT10g);
        }

        vad T1b = va_set1(pp->voffcbn) - (one - va_set1(pp->mstar)) * Vgst;
        vad T2b = T1b / T0;
        vam m2_lo = va_lt(T2b, nexp_th);
        vam m2_hi = va_gt(T2b, exp_th);
        vam m2_mid = (~m2_lo) & (~m2_hi);

        vad T3_cl = va_set1(pp->coxe_ov_cdep0) * va_sel(m2_lo, min_exp, max_exp);
        vad T9 = va_set1(pp->mstar) + T3_cl * n;
        vad dT9g = z0;
        vad dT9d = dn_dVd * T3_cl;
        vad dT9b = dn_dVb * T3_cl;
        if (va_any(m2_mid)) {
            vad ex = va_exp_clamped(T2b, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            vad T4 = va_set1(pp->coxe_ov_cdep0) * ex;
            vad T5 = T1b * T4 / T0;
            vad dg0 = va_set1(pp->coxe_ov_cdep0) * (va_set1(pp->mstar) - one) * ex / va_set1(pp->vtm);
            T9   = va_sel(m2_mid, va_set1(pp->mstar) + n * T4, T9);
            dT9b = va_sel(m2_mid, T4 * dn_dVb - dg0 * dVth_dVb - T5 * dn_dVb, dT9b);
            dT9d = va_sel(m2_mid, T4 * dn_dVd - dg0 * dVth_dVd - T5 * dn_dVd, dT9d);
            dT9g = va_sel(m2_mid, dg0 * dVgs_eff_dVg, dT9g);
        }

        vad invT9 = one / T9;
        vad invT9sq = invT9 * invT9;
        Vgsteff      = T10 * invT9;
        dVgsteff_dVg = (T9 * dT10g - T10 * dT9g) * invT9sq;
        dVgsteff_dVd = (T9 * dT10d - T10 * dT9d) * invT9sq;
        dVgsteff_dVb = (T9 * dT10b - T10 * dT9b) * invT9sq;
    }

    /* ===================================================================== */
    /* 5. Weff, Rds (rdsMod == 0), Abulk                                     */
    /* ===================================================================== */
    vad T9_phi = sqrtPhis - va_set1(pp->sqrtPhi);
    vad Weff = va_set1(pp->weff) - two * (va_set1(pp->dwg) * Vgsteff + va_set1(pp->dwb) * T9_phi);
    vad dWeff_dVg = -two * va_set1(pp->dwg);
    vad dWeff_dVb = -two * va_set1(pp->dwb) * dsqrtPhis_dVb;
    {
        vam m_w = va_lt(Weff, va_set1(2.0e-8));
        if (va_any(m_w)) {
            vad T0 = one / (va_set1(6.0e-8) - two * Weff);
            vad T0s = T0 * T0 * va_set1(4.0e-16);
            Weff = va_sel(m_w, va_set1(2.0e-8) * (va_set1(4.0e-8) - Weff) * T0, Weff);
            dWeff_dVg = va_sel(m_w, dWeff_dVg * T0s, dWeff_dVg);
            dWeff_dVb = va_sel(m_w, dWeff_dVb * T0s, dWeff_dVb);
        }
    }

    vad Rds, dRds_dVg, dRds_dVb;
    {
        vad T0 = one + va_set1(pp->prwg) * Vgsteff;
        vad invT0 = one / T0;
        vad dT0_dVg = -va_set1(pp->prwg) * invT0 * invT0;
        vad T1 = va_set1(pp->prwb) * T9_phi;
        vad dT1_dVb = va_set1(pp->prwb) * dsqrtPhis_dVb;
        vad T2 = invT0 + T1;
        vad T3 = T2 + va_sqrt(T2 * T2 + va_set1(0.01));
        vad dT3_base = one + T2 / (T3 - T2);
        Rds = va_set1(pp->rdswmin) + T3 * va_set1(pp->rds0_half);
        dRds_dVg = va_set1(pp->rds0_half) * dT3_base * dT0_dVg;
        dRds_dVb = va_set1(pp->rds0_half) * dT3_base * dT1_dVb;
    }

    vad Abulk0, dAbulk0_dVb, Abulk, dAbulk_dVg, dAbulk_dVb;
    {
        vad T9 = half * va_set1(pp->k1ox) * va_set1(pp->lpebFactor) / sqrtPhis;
        vad T1 = T9 + va_set1(pp->k2ox) - va_set1(pp->k3b) * va_set1(pp->vthNarrowW);
        vad dT1_dVb = -T9 / sqrtPhis * dsqrtPhis_dVb;

        vad T9x = va_sqrt(va_set1(pp->xj) * Xdep);
        vad tmp1 = va_set1(pp->leff) + two * T9x;
        vad T5 = va_set1(pp->leff) / tmp1;
        vad tmp2 = va_set1(pp->a0) * T5;
        vad T2 = tmp2 + va_set1(pp->b0_ov_wb1);
        vad dT2_dVb = -T9x / (tmp1 * Xdep) * dXdep_dVb;
        vad T7 = T5 * T5 * T5;

        Abulk0 = one + T1 * T2;
        dAbulk0_dVb = T1 * tmp2 * dT2_dVb + T2 * dT1_dVb;

        vad T8 = va_set1(pp->ags_a0) * T7;
        dAbulk_dVg = -T1 * T8;
        Abulk = Abulk0 + dAbulk_dVg * Vgsteff;
        dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * (dT1_dVb + va_set1(3.0) * T1 * dT2_dVb);

        vam m_a0 = va_lt(Abulk0, va_set1(0.1));
        if (va_any(m_a0)) {
            vad Ta = one / (va_set1(3.0) - va_set1(20.0) * Abulk0);
            Abulk0 = va_sel(m_a0, (va_set1(0.2) - Abulk0) * Ta, Abulk0);
            dAbulk0_dVb = va_sel(m_a0, dAbulk0_dVb * Ta * Ta, dAbulk0_dVb);
        }
        vam m_ab = va_lt(Abulk, va_set1(0.1));
        if (va_any(m_ab)) {
            vad Ta = one / (va_set1(3.0) - va_set1(20.0) * Abulk);
            vad Tasq = Ta * Ta;
            Abulk = va_sel(m_ab, (va_set1(0.2) - Abulk) * Ta, Abulk);
            dAbulk_dVb = va_sel(m_ab, dAbulk_dVb * Tasq, dAbulk_dVb);
            dAbulk_dVg = va_sel(m_ab, dAbulk_dVg * Tasq, dAbulk_dVg);
        }

        vad Tk = va_set1(pp->keta) * Vbseff;
        vam m_k = va_ge(Tk, va_set1(-0.9));
        vad T0k = one / (one + Tk);
        vad T1k = one / (va_set1(0.8) + Tk);
        vad T0 = va_sel(m_k, T0k, (va_set1(17.0) + va_set1(20.0) * Tk) * T1k);
        vad dT0_dVb = va_sel(m_k, -va_set1(pp->keta) * T0k * T0k, -va_set1(pp->keta) * T1k * T1k);
        dAbulk_dVg = dAbulk_dVg * T0;
        dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb;
        dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb;
        Abulk = Abulk * T0;
        Abulk0 = Abulk0 * T0;
    }

    /* ===================================================================== */
    /* 6. Mobility (mobMod == 0), EsatL, Vdsat, Vdseff                       */
    /* ===================================================================== */
    vad ueff, dueff_dVg, dueff_dVd, dueff_dVb;
    {
        vad T0 = Vgsteff + Vth + Vth;
        vad T2 = va_set1(pp->ua) + va_set1(pp->uc) * Vbseff;
        vad T3 = T0 * va_set1(pp->invToxe);
        vad T5, dDen_dVg, dDen_dVd, dDen_dVb;
        if (!pp->has_ud) {
            T5 = T3 * (T2 + va_set1(pp->ub) * T3);
            dDen_dVg = (T2 + two * va_set1(pp->ub) * T3) * va_set1(pp->invToxe);
            dDen_dVd = dDen_dVg * two * dVth_dVd;
            dDen_dVb = dDen_dVg * two * dVth_dVb + va_set1(pp->uc) * T3;
        } else if (!pp->is_v6) {
            vad invT3 = one / T3;
            vad T6 = va_set1(pp->ud) * invT3 * invT3 * Vth * Vth;
            T5 = T3 * (T2 + va_set1(pp->ub) * T3) + T6;
            vad T7 = -two * T6 / T0;
            dDen_dVg = (T2 + two * va_set1(pp->ub) * T3) * va_set1(pp->invToxe) + T7;
            dDen_dVd = dDen_dVg * two * dVth_dVd;
            dDen_dVb = dDen_dVg * two * dVth_dVb + va_set1(pp->uc) * T3;
        } else {
            vad T12 = va_sqrt(Vth * Vth + va_set1(0.0001));
            vad T9 = one / (Vgsteff + two * T12);
            vad T10 = T9 * va_set1(pp->toxe);
            vad T8 = va_set1(pp->ud) * T10 * T10 * Vth;
            vad T6 = T8 * Vth;
            T5 = T3 * (T2 + va_set1(pp->ub) * T3) + T6;
            vad T7 = -two * T6 * T9;
            vad T11 = T7 * Vth / T12;
            vad base_g = (T2 + two * va_set1(pp->ub) * T3) * va_set1(pp->invToxe);
            vad T13 = two * (base_g + T11 + T8);
            dDen_dVd = T13 * dVth_dVd;
            dDen_dVb = T13 * dVth_dVb + va_set1(pp->uc) * T3;
            dDen_dVg = base_g + T7;
        }

        vam m_den = va_ge(T5, va_set1(-0.8));
        vad T9 = one / (va_set1(7.0) + va_set1(10.0) * T5);
        vad Denomi = va_sel(m_den, one + T5, (va_set1(0.6) + T5) * T9);
        vad T9sq = T9 * T9;
        dDen_dVg = va_sel(m_den, dDen_dVg, dDen_dVg * T9sq);
        dDen_dVd = va_sel(m_den, dDen_dVd, dDen_dVd * T9sq);
        dDen_dVb = va_sel(m_den, dDen_dVb, dDen_dVb * T9sq);

        vad invDen = one / Denomi;
        ueff = va_set1(pp->u0temp) * invDen;
        vad m_u_inv = -ueff * invDen;
        dueff_dVg = m_u_inv * dDen_dVg;
        dueff_dVd = m_u_inv * dDen_dVd;
        dueff_dVb = m_u_inv * dDen_dVb;
    }

    vad WVCox = Weff * va_set1(pp->vsattemp) * va_set1(pp->coxe);
    vad WVCoxRds = WVCox * Rds;

    vad Esat = two * va_set1(pp->vsattemp) / ueff;
    vad EsatL = Esat * va_set1(pp->leff);
    vad T0_es = -EsatL / ueff;
    vad dEsatL_dVg = T0_es * dueff_dVg;
    vad dEsatL_dVd = T0_es * dueff_dVd;
    vad dEsatL_dVb = T0_es * dueff_dVb;

    vad Lambda = va_set1(pp->a2), dLambda_dVg = z0;
    if (pp->f_has_a1) {
        vam m_pos = va_gt(va_set1(pp->a1), z0);
        vam m_neg = va_lt(va_set1(pp->a1), z0);
        if (va_any(m_pos)) {
            vad T0 = one - va_set1(pp->a2);
            vad T1 = T0 - va_set1(pp->a1) * Vgsteff - va_set1(0.0001);
            vad T2 = va_sqrt(va_max(T1 * T1 + va_set1(0.0004) * T0, va_set1(1.0e-300)));
            Lambda = va_sel(m_pos, va_set1(pp->a2) + T0 - half * (T1 + T2), Lambda);
            dLambda_dVg = va_sel(m_pos, half * va_set1(pp->a1) * (one + T1 / T2), dLambda_dVg);
        }
        if (va_any(m_neg)) {
            vad T1 = va_set1(pp->a2) + va_set1(pp->a1) * Vgsteff - va_set1(0.0001);
            vad T2 = va_sqrt(va_max(T1 * T1 + va_set1(0.0004) * va_set1(pp->a2), va_set1(1.0e-300)));
            Lambda = va_sel(m_neg, half * (T1 + T2), Lambda);
            dLambda_dVg = va_sel(m_neg, half * va_set1(pp->a1) * (one + T1 / T2), dLambda_dVg);
        }
    }

    vad Vgst2Vtm = Vgsteff + two * va_set1(pp->vtm);
    vad invWeff = one / Weff;
    vam m_rds_pos = va_gt(Rds, z0);
    vad invRds_safe = va_sel(m_rds_pos, one / va_max(Rds, va_set1(1.0e-300)), z0);
    vad tmp2_rw = dRds_dVg * invRds_safe + dWeff_dVg * invWeff;
    vad tmp3_rw = dRds_dVb * invRds_safe + dWeff_dVb * invWeff;

    vad Vdsat, dVdsat_dVg, dVdsat_dVd, dVdsat_dVb, tmp1_lam;
    {
        vam m_simple = va_eq(Rds, z0) & va_eq(Lambda, one);
        if (va_all(m_simple)) {
            vad T0 = one / (Abulk * EsatL + Vgst2Vtm);
            tmp1_lam = z0;
            vad T1 = T0 * T0;
            vad T2 = Vgst2Vtm * T0;
            vad T3 = EsatL * Vgst2Vtm;
            Vdsat = T3 * T0;
            vad dT0_dVg = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + one) * T1;
            vad dT0_dVd = -(Abulk * dEsatL_dVd) * T1;
            vad dT0_dVb = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * T1;
            dVdsat_dVg = T3 * dT0_dVg + T2 * dEsatL_dVg + EsatL * T0;
            dVdsat_dVd = T3 * dT0_dVd + T2 * dEsatL_dVd;
            dVdsat_dVb = T3 * dT0_dVb + T2 * dEsatL_dVb;
        } else {
            vad invLam = one / Lambda;
            vad twoInvLamM1 = two * invLam - one;
            vad invLamM1 = invLam - one;
            tmp1_lam = dLambda_dVg * invLam * invLam;
            vad T9 = Abulk * WVCoxRds;
            vad T8 = Abulk * T9;
            vad T7 = Vgst2Vtm * T9;
            vad T6 = Vgst2Vtm * WVCoxRds;
            vad T0 = two * Abulk * (T9 - one + invLam);
            vad dT0_dVg = two * (T8 * tmp2_rw - Abulk * tmp1_lam + (two * T9 + invLamM1) * dAbulk_dVg);
            vad dT0_dVb = two * (T8 * (two / Abulk * dAbulk_dVb + tmp3_rw) + invLamM1 * dAbulk_dVb);
            vad T1 = Vgst2Vtm * twoInvLamM1 + Abulk * EsatL + va_set1(3.0) * T7;
            vad dT1_dVg = twoInvLamM1 - two * Vgst2Vtm * tmp1_lam
                        + Abulk * dEsatL_dVg + EsatL * dAbulk_dVg
                        + va_set1(3.0) * (T9 + T7 * tmp2_rw + T6 * dAbulk_dVg);
            vad dT1_dVb = Abulk * dEsatL_dVb + EsatL * dAbulk_dVb
                        + va_set1(3.0) * (T6 * dAbulk_dVb + T7 * tmp3_rw);
            vad dT1_dVd = Abulk * dEsatL_dVd;
            vad T2 = Vgst2Vtm * (EsatL + two * T6);
            vad dT2_dVg = EsatL + Vgst2Vtm * dEsatL_dVg + T6 * (va_set1(4.0) + two * Vgst2Vtm * tmp2_rw);
            vad dT2_dVb = Vgst2Vtm * (dEsatL_dVb + two * T6 * tmp3_rw);
            vad dT2_dVd = Vgst2Vtm * dEsatL_dVd;
            vad T3 = va_sqrt(va_max(T1 * T1 - two * T0 * T2, va_set1(1.0e-300)));
            vad invT0 = one / va_sel(m_simple, one, T0);
            vad invT3 = one / T3;
            Vdsat = (T1 - T3) * invT0;
            dVdsat_dVg = (dT1_dVg - (T1 * dT1_dVg - dT0_dVg * T2 - T0 * dT2_dVg) * invT3 - Vdsat * dT0_dVg) * invT0;
            dVdsat_dVb = (dT1_dVb - (T1 * dT1_dVb - dT0_dVb * T2 - T0 * dT2_dVb) * invT3 - Vdsat * dT0_dVb) * invT0;
            dVdsat_dVd = (dT1_dVd - (T1 * dT1_dVd - T0 * dT2_dVd) * invT3) * invT0;
            if (va_any(m_simple)) {
                vad sT0 = one / (Abulk * EsatL + Vgst2Vtm);
                vad sT1 = sT0 * sT0;
                vad sT2 = Vgst2Vtm * sT0;
                vad sT3 = EsatL * Vgst2Vtm;
                vad sdT0g = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + one) * sT1;
                vad sdT0d = -(Abulk * dEsatL_dVd) * sT1;
                vad sdT0b = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * sT1;
                Vdsat      = va_sel(m_simple, sT3 * sT0, Vdsat);
                dVdsat_dVg = va_sel(m_simple, sT3 * sdT0g + sT2 * dEsatL_dVg + EsatL * sT0, dVdsat_dVg);
                dVdsat_dVd = va_sel(m_simple, sT3 * sdT0d + sT2 * dEsatL_dVd, dVdsat_dVd);
                dVdsat_dVb = va_sel(m_simple, sT3 * sdT0b + sT2 * dEsatL_dVb, dVdsat_dVb);
                tmp1_lam   = va_sel(m_simple, z0, tmp1_lam);
            }
        }
    }

    /* Vdseff */
    vad Vdseff, dVdseff_dVg, dVdseff_dVd, dVdseff_dVb;
    {
        vad T1 = Vdsat - Vds - va_set1(pp->delta);
        vad dT1_dVg = dVdsat_dVg;
        vad dT1_dVd = dVdsat_dVd - one;
        vad dT1_dVb = dVdsat_dVb;
        vad T2 = va_sqrt(T1 * T1 + va_set1(4.0) * va_set1(pp->delta) * Vdsat);
        vad invT2 = one / T2;
        vad T0 = T1 * invT2;
        vad T9 = two * va_set1(pp->delta);
        vad T3 = T9 * invT2;
        vad dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
        vad dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
        vad dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;

        vam m_ge = va_ge(T1, z0);
        vad ve_p = Vdsat - half * (T1 + T2);
        vad dg_p = dVdsat_dVg - half * (dT1_dVg + dT2_dVg);
        vad dd_p = dVdsat_dVd - half * (dT1_dVd + dT2_dVd);
        vad db_p = dVdsat_dVb - half * (dT1_dVb + dT2_dVb);

        vad inv_diff = one / (T2 - T1);
        vad T4 = T9 * inv_diff;
        vad T5 = one - T4;
        vad T6 = Vdsat * T4 * inv_diff;
        vad ve_n = Vdsat * T5;
        vad dg_n = dVdsat_dVg * T5 + T6 * (dT2_dVg - dT1_dVg);
        vad dd_n = dVdsat_dVd * T5 + T6 * (dT2_dVd - dT1_dVd);
        vad db_n = dVdsat_dVb * T5 + T6 * (dT2_dVb - dT1_dVb);

        Vdseff      = va_sel(m_ge, ve_p, ve_n);
        dVdseff_dVg = va_sel(m_ge, dg_p, dg_n);
        dVdseff_dVd = va_sel(m_ge, dd_p, dd_n);
        dVdseff_dVb = va_sel(m_ge, db_p, db_n);

        vam m_vds0 = va_eq(Vds, z0);
        Vdseff      = va_sel(m_vds0, z0, Vdseff);
        dVdseff_dVg = va_sel(m_vds0, z0, dVdseff_dVg);
        dVdseff_dVb = va_sel(m_vds0, z0, dVdseff_dVb);
        Vdseff      = va_min(Vdseff, Vds);
    }
    vad diffVds = Vds - Vdseff;

    /* Velocity Overshoot (lambda > 0) */
    if (pp->f_lambda) {
        vad T1 = va_set1(pp->leff) * ueff;
        vad T2 = va_set1(pp->lambda) / T1;
        vad T3 = -T2 / T1 * va_set1(pp->leff);
        vad dT2_dVd = T3 * dueff_dVd;
        vad dT2_dVg = T3 * dueff_dVg;
        vad dT2_dVb = T3 * dueff_dVb;
        vad T5 = one / (Esat * va_set1(pp->litl));
        vad T4 = -T5 / EsatL;
        vad dT5_dVg = dEsatL_dVg * T4;
        vad dT5_dVd = dEsatL_dVd * T4;
        vad dT5_dVb = dEsatL_dVb * T4;
        vad T6 = one + diffVds * T5;
        vad dT6_dVg = dT5_dVg * diffVds - dVdseff_dVg * T5;
        vad dT6_dVd = dT5_dVd * diffVds + (one - dVdseff_dVd) * T5;
        vad dT6_dVb = dT5_dVb * diffVds - dVdseff_dVb * T5;
        vad T7 = two / (T6 * T6 + one);
        vad T8 = one - T7;
        vad T9 = T6 * T7 * T7;
        vad T10 = one + T2 * T8;
        vam m_eq1 = va_eq(T10, one);
        vad dT10_dVg = va_sel(m_eq1, z0, dT2_dVg * T8 + T2 * T9 * dT6_dVg);
        vad dT10_dVd = va_sel(m_eq1, z0, dT2_dVd * T8 + T2 * T9 * dT6_dVd);
        vad dT10_dVb = va_sel(m_eq1, z0, dT2_dVb * T8 + T2 * T9 * dT6_dVb);
        dEsatL_dVg = va_sel((pp->f_lambda ? va_mask_all() : va_mask_zero()), dEsatL_dVg * T10 + EsatL * dT10_dVg, dEsatL_dVg);
        dEsatL_dVd = va_sel((pp->f_lambda ? va_mask_all() : va_mask_zero()), dEsatL_dVd * T10 + EsatL * dT10_dVd, dEsatL_dVd);
        dEsatL_dVb = va_sel((pp->f_lambda ? va_mask_all() : va_mask_zero()), dEsatL_dVb * T10 + EsatL * dT10_dVb, dEsatL_dVb);
        EsatL      = va_sel((pp->f_lambda ? va_mask_all() : va_mask_zero()), EsatL * T10, EsatL);
        if (pp->is_v6)
            Esat = va_sel((pp->f_lambda ? va_mask_all() : va_mask_zero()), EsatL * va_set1(pp->invLeff), Esat);
    }

    /* ===================================================================== */

    p->vds = vds;
    p->vgs = vgs;
    p->vbs = vbs;
    p->vbd = vbd;
    p->vgd = vgd;
    p->vgb = vgb;
    p->Vds = Vds;
    p->Vgs_eff = Vgs_eff;
    p->dVgs_eff_dVg = dVgs_eff_dVg;
    p->vgs_eff = vgs_eff;
    p->dvgs_eff_dvg = dvgs_eff_dvg;
    p->vgd_eff = vgd_eff;
    p->dvgd_eff_dvg = dvgd_eff_dvg;
    p->Vbseff = Vbseff;
    p->dVbseff_dVb = dVbseff_dVb;
    p->Phis = Phis;
    p->Vth = Vth;
    p->dVth_dVd = dVth_dVd;
    p->dVth_dVb = dVth_dVb;
    p->n = n;
    p->dn_dVd = dn_dVd;
    p->dn_dVb = dn_dVb;
    p->Vgst = Vgst;
    p->Vgsteff = Vgsteff;
    p->dVgsteff_dVg = dVgsteff_dVg;
    p->dVgsteff_dVd = dVgsteff_dVd;
    p->dVgsteff_dVb = dVgsteff_dVb;
    p->Weff = Weff;
    p->dWeff_dVg = dWeff_dVg;
    p->dWeff_dVb = dWeff_dVb;
    p->Rds = Rds;
    p->dRds_dVg = dRds_dVg;
    p->dRds_dVb = dRds_dVb;
    p->Abulk = Abulk;
    p->dAbulk_dVg = dAbulk_dVg;
    p->dAbulk_dVb = dAbulk_dVb;
    p->Abulk0 = Abulk0;
    p->dAbulk0_dVb = dAbulk0_dVb;
    p->ueff = ueff;
    p->dueff_dVg = dueff_dVg;
    p->dueff_dVd = dueff_dVd;
    p->dueff_dVb = dueff_dVb;
    p->Esat = Esat;
    p->EsatL = EsatL;
    p->dEsatL_dVg = dEsatL_dVg;
    p->dEsatL_dVd = dEsatL_dVd;
    p->dEsatL_dVb = dEsatL_dVb;
    p->Vdsat = Vdsat;
    p->dVdsat_dVg = dVdsat_dVg;
    p->dVdsat_dVd = dVdsat_dVd;
    p->dVdsat_dVb = dVdsat_dVb;
    p->Vdseff = Vdseff;
    p->dVdseff_dVg = dVdseff_dVg;
    p->dVdseff_dVd = dVdseff_dVd;
    p->dVdseff_dVb = dVdseff_dVb;
    p->WVCoxRds = WVCoxRds;
    p->Lambda = Lambda;
    p->diffVds = diffVds;
    p->Vgst2Vtm = Vgst2Vtm;
    p->tmp1_lam = tmp1_lam;
    p->tmp2_rw = tmp2_rw;
    p->tmp3_rw = tmp3_rw;
    p->m_fwd = m_fwd;
}

__attribute__((noinline))
static void
bsim4_apple_p2_ids_diode(
    const BSIM4ApplePParam * restrict pp,
    const BSIM4ApplePipe * restrict p,
    const BSIM4AppleJct * const jpts[B4A_W],
    BSIM4AppleCheck * const cpts[B4A_W],
    BSIM4AppleStamp * const spts[B4A_W],
    BSIM4AppleL2 * const l2pts[B4A_W],
    int nact,
    double ckt_gmin,
    BSIM4AppleOut * restrict out,
    int need_aos)
{
    const vad z0 = va_set1(0.0);
    const vad one = va_set1(1.0);
    const vad half = va_set1(0.5);
    const vad two = va_set1(2.0);
    const vad gmin = va_set1(ckt_gmin);
    const vad max_exp = va_set1(B4V_MAX_EXP);
    const vad min_exp = va_set1(B4V_MIN_EXP);
    const vad exp_th = va_set1(B4V_EXP_THRESHOLD);
    const vad nexp_th = va_set1(-B4V_EXP_THRESHOLD);
    const vad epssi = va_set1(B4V_EPSSI);
    (void)nexp_th;

    uint16_t jct_any_flags = 0;
    double a_m[B4A_W], a_dc[B4A_W], a_sc[B4A_W];
    for (int l = 0; l < B4A_W; l++) {
        jct_any_flags |= jpts[l]->flags;
        a_m[l]  = spts[l]->m;
        a_dc[l] = spts[l]->drainCond;
        a_sc[l] = spts[l]->sourceCond;
    }
    const vad m_v          = va_load(a_m);
    const vad drainCond_v  = va_load(a_dc);
    const vad sourceCond_v = va_load(a_sc);

    const vad vds = p->vds;
    const vad vgs = p->vgs;
    const vad vbs = p->vbs;
    const vad vbd = p->vbd;
    const vad vgd = p->vgd;
    const vad Vds = p->Vds;
    const vad vgs_eff = p->vgs_eff;
    const vad dvgs_eff_dvg = p->dvgs_eff_dvg;
    const vad vgd_eff = p->vgd_eff;
    const vad dvgd_eff_dvg = p->dvgd_eff_dvg;
    const vad Vbseff = p->Vbseff;
    const vad dVbseff_dVb = p->dVbseff_dVb;
    const vad Vgsteff = p->Vgsteff;
    const vad dVgsteff_dVg = p->dVgsteff_dVg;
    const vad dVgsteff_dVd = p->dVgsteff_dVd;
    const vad dVgsteff_dVb = p->dVgsteff_dVb;
    const vad Weff = p->Weff;
    const vad dWeff_dVg = p->dWeff_dVg;
    const vad dWeff_dVb = p->dWeff_dVb;
    const vad Rds = p->Rds;
    const vad dRds_dVg = p->dRds_dVg;
    const vad dRds_dVb = p->dRds_dVb;
    const vad Abulk = p->Abulk;
    const vad dAbulk_dVg = p->dAbulk_dVg;
    const vad dAbulk_dVb = p->dAbulk_dVb;
    const vad ueff = p->ueff;
    const vad dueff_dVg = p->dueff_dVg;
    const vad dueff_dVd = p->dueff_dVd;
    const vad dueff_dVb = p->dueff_dVb;
    const vad Esat = p->Esat;
    const vad EsatL = p->EsatL;
    const vad dEsatL_dVg = p->dEsatL_dVg;
    const vad dEsatL_dVd = p->dEsatL_dVd;
    const vad dEsatL_dVb = p->dEsatL_dVb;
    const vad Vdsat = p->Vdsat;
    const vad dVdsat_dVg = p->dVdsat_dVg;
    const vad dVdsat_dVd = p->dVdsat_dVd;
    const vad dVdsat_dVb = p->dVdsat_dVb;
    const vad Vdseff = p->Vdseff;
    const vad dVdseff_dVg = p->dVdseff_dVg;
    const vad dVdseff_dVd = p->dVdseff_dVd;
    const vad dVdseff_dVb = p->dVdseff_dVb;
    const vad WVCoxRds = p->WVCoxRds;
    const vad Lambda = p->Lambda;
    const vad diffVds = p->diffVds;
    const vad Vgst2Vtm = p->Vgst2Vtm;
    const vad tmp1_lam = p->tmp1_lam;
    const vad tmp2_rw = p->tmp2_rw;
    const vad tmp3_rw = p->tmp3_rw;
    const vam m_fwd = p->m_fwd;

    /* 7. Vasat, Idl, CLM, DIBL, DITS, SCBE, Isub, Ids, Fsevl                */
    /* ===================================================================== */
    vad Vasat, dVasat_dVg, dVasat_dVb, dVasat_dVd;
    {
        vad invVgst2Vtm = one / Vgst2Vtm;
        vad tmp4 = one - half * Abulk * Vdsat * invVgst2Vtm;
        vad T9 = WVCoxRds * Vgsteff;
        vad T8 = T9 * invVgst2Vtm;
        vad T0 = EsatL + Vdsat + two * T9 * tmp4;
        vad T7 = two * WVCoxRds * tmp4;
        vad dT0_dVg = dEsatL_dVg + dVdsat_dVg + T7 * (one + tmp2_rw * Vgsteff)
                    - T8 * (Abulk * dVdsat_dVg - Abulk * Vdsat * invVgst2Vtm + Vdsat * dAbulk_dVg);
        vad dT0_dVb = dEsatL_dVb + dVdsat_dVb + T7 * tmp3_rw * Vgsteff
                    - T8 * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
        vad dT0_dVd = dEsatL_dVd + dVdsat_dVd - T8 * Abulk * dVdsat_dVd;
        vad T9b = WVCoxRds * Abulk;
        vad T1 = two / Lambda - one + T9b;
        vad dT1_dVg = -two * tmp1_lam + WVCoxRds * (Abulk * tmp2_rw + dAbulk_dVg);
        vad dT1_dVb = dAbulk_dVb * WVCoxRds + T9b * tmp3_rw;
        vad invT1 = one / T1;
        Vasat = T0 * invT1;
        dVasat_dVg = (dT0_dVg - Vasat * dT1_dVg) * invT1;
        dVasat_dVb = (dT0_dVb - Vasat * dT1_dVb) * invT1;
        dVasat_dVd = dT0_dVd * invT1;
    }

    /* Idl & CoxeffWovL */
    vad CoxeffWovL, Idl, dIdl_dVg, dIdl_dVd, dIdl_dVb;
    {
        vad T0 = (Vgsteff + va_set1(pp->vtfbphi2)) * va_set1(pp->inv_2e8_toxp);
        vad tmp3 = va_exp(va_set1(pp->bdos0p7) * va_log_pos(T0));
        vad T1 = one + tmp3;
        vad invT1 = one / T1;
        vad T2 = va_set1(pp->bdos0p7) * tmp3 / T0;
        vad Tcen = va_set1(pp->ados1p9) * invT1;
        vad dTcen_dVg = -Tcen * T2 * va_set1(pp->inv_2e8_toxp) * invT1;
        vad Coxeff = epssi * va_set1(pp->coxp) / (epssi + va_set1(pp->coxp) * Tcen);
        vad dCoxeff_dVg = -Coxeff * Coxeff * dTcen_dVg * va_set1(1.0 / B4V_EPSSI);
        CoxeffWovL = Coxeff * Weff * va_set1(pp->invLeff);
        vad beta = ueff * CoxeffWovL;
        vad T3 = ueff * va_set1(pp->invLeff);
        vad dbeta_dVg = CoxeffWovL * dueff_dVg + T3 * (Weff * dCoxeff_dVg + Coxeff * dWeff_dVg);
        vad dbeta_dVd = CoxeffWovL * dueff_dVd;
        vad dbeta_dVb = CoxeffWovL * dueff_dVb + T3 * Coxeff * dWeff_dVb;

        vad invVgst2Vtm = one / Vgst2Vtm;
        vad T0a = one - half * Vdseff * Abulk * invVgst2Vtm;
        vad dT0a_dVg = -half * (Abulk * dVdseff_dVg - Abulk * Vdseff * invVgst2Vtm + Vdseff * dAbulk_dVg) * invVgst2Vtm;
        vad dT0a_dVd = -half * Abulk * dVdseff_dVd * invVgst2Vtm;
        vad dT0a_dVb = -half * (Abulk * dVdseff_dVb + dAbulk_dVb * Vdseff) * invVgst2Vtm;

        vad fgche1 = Vgsteff * T0a;
        vad dfgche1_dVg = Vgsteff * dT0a_dVg + T0a;
        vad dfgche1_dVd = Vgsteff * dT0a_dVd;
        vad dfgche1_dVb = Vgsteff * dT0a_dVb;

        vad invEsatL = one / EsatL;
        vad T9 = Vdseff * invEsatL;
        vad fgche2 = one + T9;
        vad invFg2 = one / fgche2;
        vad dfgche2_dVg = (dVdseff_dVg - T9 * dEsatL_dVg) * invEsatL;
        vad dfgche2_dVd = (dVdseff_dVd - T9 * dEsatL_dVd) * invEsatL;
        vad dfgche2_dVb = (dVdseff_dVb - T9 * dEsatL_dVb) * invEsatL;

        vad gche = beta * fgche1 * invFg2;
        vad dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg - gche * dfgche2_dVg) * invFg2;
        vad dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd - gche * dfgche2_dVd) * invFg2;
        vad dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb - gche * dfgche2_dVb) * invFg2;

        vad T0g = one + gche * Rds;
        vad invT0g = one / T0g;
        Idl = gche * invT0g;
        vad T1g = (one - Idl * Rds) * invT0g;
        vad T2g = Idl * Idl;
        dIdl_dVg = T1g * dgche_dVg - T2g * dRds_dVg;
        dIdl_dVd = T1g * dgche_dVd;
        dIdl_dVb = T1g * dgche_dVb - T2g * dRds_dVb;
    }

    /* FP (pocket degradation) */
    vad FP = one, dFP_dVg = z0;
    if (pp->f_fprout) {
        vad T9 = va_set1(pp->fprout) * va_set1(pp->sqrtLeff) / Vgst2Vtm;
        vad fp = one / (one + T9);
        FP = va_sel((pp->f_fprout ? va_mask_all() : va_mask_zero()), fp, one);
        dFP_dVg = va_sel((pp->f_fprout ? va_mask_all() : va_mask_zero()), fp * fp * T9 / Vgst2Vtm, z0);
    }

    /* PvagTerm */
    vad PvagTerm, dPvagTerm_dVg, dPvagTerm_dVb, dPvagTerm_dVd;
    {
        vad invEsatL = one / EsatL;
        vad T8 = va_set1(pp->pvag) * invEsatL;
        vad T9 = T8 * Vgsteff;
        vam m_gt = va_gt(T9, va_set1(-0.9));
        vad T4 = one / (va_set1(17.0) + va_set1(20.0) * T9);
        vad T4sq = T4 * T4;
        vad scale_d = va_sel(m_gt, one, T4sq);
        PvagTerm = va_sel(m_gt, one + T9, (va_set1(0.8) + T9) * T4);
        dPvagTerm_dVg = T8 * (one - Vgsteff * dEsatL_dVg * invEsatL) * scale_d;
        vad f_bd = -T9 * invEsatL * scale_d;
        dPvagTerm_dVb = f_bd * dEsatL_dVb;
        dPvagTerm_dVd = f_bd * dEsatL_dVd;
    }

    /* VACLM & Cclm */
    vad Cclm = max_exp, dCclm_dVg = z0, dCclm_dVb = z0, dCclm_dVd = z0;
    vad VACLM = max_exp, dVACLM_dVg = z0, dVACLM_dVb = z0, dVACLM_dVd = z0;
    {
        vam m_clm = (pp->f_pclm ? va_mask_all() : va_mask_zero()) & va_gt(diffVds, va_set1(1.0e-10));
        if (va_any(m_clm)) {
            vad T0 = one + Rds * Idl;
            vad dT0_dVg = dRds_dVg * Idl + Rds * dIdl_dVg;
            vad dT0_dVd = Rds * dIdl_dVd;
            vad dT0_dVb = dRds_dVb * Idl + Rds * dIdl_dVb;
            vad invEsat = one / Esat;
            vad T2 = Vdsat * invEsat;
            vad T1 = va_set1(pp->leff) + T2;
            vad dT1_dVg = (dVdsat_dVg - T2 * dEsatL_dVg * va_set1(pp->invLeff)) * invEsat;
            vad dT1_dVd = (dVdsat_dVd - T2 * dEsatL_dVd * va_set1(pp->invLeff)) * invEsat;
            vad dT1_dVb = (dVdsat_dVb - T2 * dEsatL_dVb * va_set1(pp->invLeff)) * invEsat;
            vad cc = FP * PvagTerm * T0 * T1 * va_set1(pp->inv_pclm_litl);
            vad invPv = one / PvagTerm, invT0 = one / T0, invT1 = one / T1;
            vad dcc_g = cc * (dFP_dVg / FP + dPvagTerm_dVg * invPv + dT0_dVg * invT0 + dT1_dVg * invT1);
            vad dcc_b = cc * (dPvagTerm_dVb * invPv + dT0_dVb * invT0 + dT1_dVb * invT1);
            vad dcc_d = cc * (dPvagTerm_dVd * invPv + dT0_dVd * invT0 + dT1_dVd * invT1);
            Cclm = va_sel(m_clm, cc, max_exp);
            dCclm_dVg = va_sel(m_clm, dcc_g, z0);
            dCclm_dVb = va_sel(m_clm, dcc_b, z0);
            dCclm_dVd = va_sel(m_clm, dcc_d, z0);
            VACLM = va_sel(m_clm, cc * diffVds, max_exp);
            dVACLM_dVg = va_sel(m_clm, dcc_g * diffVds - dVdseff_dVg * cc, z0);
            dVACLM_dVb = va_sel(m_clm, dcc_b * diffVds - dVdseff_dVb * cc, z0);
            dVACLM_dVd = va_sel(m_clm, dcc_d * diffVds + (one - dVdseff_dVd) * cc, z0);
        }
    }

    /* VADIBL */
    vad VADIBL = max_exp, dVADIBL_dVg = z0, dVADIBL_dVb = z0, dVADIBL_dVd = z0;
    if (pp->f_thetaRout) {
        vad T8 = Abulk * Vdsat;
        vad T0 = Vgst2Vtm * T8;
        vad dT0_dVg = Vgst2Vtm * Abulk * dVdsat_dVg + T8 + Vgst2Vtm * Vdsat * dAbulk_dVg;
        vad dT0_dVb = Vgst2Vtm * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
        vad dT0_dVd = Vgst2Vtm * Abulk * dVdsat_dVd;
        vad T1 = Vgst2Vtm + T8;
        vad dT1_dVg = one + Abulk * dVdsat_dVg + Vdsat * dAbulk_dVg;
        vad dT1_dVb = Abulk * dVdsat_dVb + dAbulk_dVb * Vdsat;
        vad dT1_dVd = Abulk * dVdsat_dVd;
        vad invT1 = one / T1;
        vad invT1sq = invT1 * invT1;
        vad va = (Vgst2Vtm - T0 * invT1) * va_set1(pp->inv_thetaRout);
        vad dva_g = (one - dT0_dVg * invT1 + T0 * dT1_dVg * invT1sq) * va_set1(pp->inv_thetaRout);
        vad dva_b = (-dT0_dVb * invT1 + T0 * dT1_dVb * invT1sq) * va_set1(pp->inv_thetaRout);
        vad dva_d = (-dT0_dVd * invT1 + T0 * dT1_dVd * invT1sq) * va_set1(pp->inv_thetaRout);

        vad T7 = va_set1(pp->pdiblb) * Vbseff;
        vam m_ge = va_ge(T7, va_set1(-0.9));
        vad T3_p = one / (one + T7);
        vad T4_n = one / (va_set1(0.8) + T7);
        vad T3_n = (va_set1(17.0) + va_set1(20.0) * T7) * T4_n;
        vad T3 = va_sel(m_ge, T3_p, T3_n);
        vad va_p = va * T3_p;
        vad dvab_p = (dva_b - va_p * va_set1(pp->pdiblb)) * T3_p;
        vad dvab_n = dva_b * T3_n - va * va_set1(pp->pdiblb) * T4_n * T4_n;
        va = va * T3;
        dva_g = dva_g * T3;
        dva_d = dva_d * T3;
        dva_b = va_sel(m_ge, dvab_p, dvab_n);

        dva_g = dva_g * PvagTerm + va * dPvagTerm_dVg;
        dva_b = dva_b * PvagTerm + va * dPvagTerm_dVb;
        dva_d = dva_d * PvagTerm + va * dPvagTerm_dVd;
        va = va * PvagTerm;

        VADIBL      = va_sel((pp->f_thetaRout ? va_mask_all() : va_mask_zero()), va, max_exp);
        dVADIBL_dVg = va_sel((pp->f_thetaRout ? va_mask_all() : va_mask_zero()), dva_g, z0);
        dVADIBL_dVb = va_sel((pp->f_thetaRout ? va_mask_all() : va_mask_zero()), dva_b, z0);
        dVADIBL_dVd = va_sel((pp->f_thetaRout ? va_mask_all() : va_mask_zero()), dva_d, z0);
    }

    vad Va = Vasat + VACLM;
    vad dVa_dVg = dVasat_dVg + dVACLM_dVg;
    vad dVa_dVb = dVasat_dVb + dVACLM_dVb;
    vad dVa_dVd = dVasat_dVd + dVACLM_dVd;

    /* VADITS (only compute exp when pdits > MIN_EXP!) */
    vad VADITS = max_exp, dVADITS_dVg = z0, dVADITS_dVd = z0;
    if (pp->f_pdits) {
        vad T0 = va_set1(pp->pditsd) * Vds;
        vam m_le = va_le(T0, exp_th) & (pp->f_pdits ? va_mask_all() : va_mask_zero());
        vad T1 = max_exp, dT1_dVd = z0;
        if (va_any(m_le)) {
            vad ex = va_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            T1 = va_sel(m_le, ex, max_exp);
            dT1_dVd = va_sel(m_le, ex * va_set1(pp->pditsd), z0);
        }
        vad vd0 = (one + va_set1(pp->pditsl_term) * T1) * va_set1(pp->inv_pdits);
        VADITS      = va_sel((pp->f_pdits ? va_mask_all() : va_mask_zero()), vd0 * FP, max_exp);
        dVADITS_dVg = va_sel((pp->f_pdits ? va_mask_all() : va_mask_zero()), vd0 * dFP_dVg, z0);
        dVADITS_dVd = va_sel((pp->f_pdits ? va_mask_all() : va_mask_zero()), FP * va_set1(pp->pditsl_term) * dT1_dVd * va_set1(pp->inv_pdits), z0);
    }

    /* VASCBE */
    vad VASCBE = max_exp, dVASCBE_dVg = z0, dVASCBE_dVd = z0, dVASCBE_dVb = z0;
    if (pp->f_pscbe) {
        vad scbe_hi = max_exp * va_set1(pp->leff) * va_set1(pp->inv_pscbe2);
        vad thr = va_set1(pp->pscbe1_litl) * va_set1(1.0 / B4V_EXP_THRESHOLD);
        vam m_act = (pp->f_pscbe ? va_mask_all() : va_mask_zero()) & va_gt(diffVds, thr);
        VASCBE = va_sel((pp->f_pscbe ? va_mask_all() : va_mask_zero()), scbe_hi, max_exp);
        if (va_any(m_act)) {
            vad diff_safe = va_max(diffVds, va_set1(1.0e-30));
            vad T0 = va_set1(pp->pscbe1_litl) / diff_safe;
            vad vs = va_set1(pp->leff) * va_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD) * va_set1(pp->inv_pscbe2);
            vad T1 = T0 * vs / diff_safe;
            VASCBE      = va_sel(m_act, vs, VASCBE);
            dVASCBE_dVg = va_sel(m_act, T1 * dVdseff_dVg, z0);
            dVASCBE_dVd = va_sel(m_act, -T1 * (one - dVdseff_dVd), z0);
            dVASCBE_dVb = va_sel(m_act, T1 * dVdseff_dVb, z0);
        }
    }

    /* Add DIBL */
    vad invVADIBL = one / VADIBL;
    vad T9_dibl = diffVds * invVADIBL;
    vad T0_dibl = one + T9_dibl;
    vad Idsa = Idl * T0_dibl;
    vad dIdsa_dVg = T0_dibl * dIdl_dVg - Idl * (dVdseff_dVg + T9_dibl * dVADIBL_dVg) * invVADIBL;
    vad dIdsa_dVd = T0_dibl * dIdl_dVd + Idl * (one - dVdseff_dVd - T9_dibl * dVADIBL_dVd) * invVADIBL;
    vad dIdsa_dVb = T0_dibl * dIdl_dVb - Idl * (dVdseff_dVb + T9_dibl * dVADIBL_dVb) * invVADIBL;

    /* Add DITS */
    {
        vad invVADITS = one / VADITS;
        vad T9 = diffVds * invVADITS;
        vad T0 = one + T9;
        dIdsa_dVg = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVADITS_dVg) * invVADITS;
        dIdsa_dVd = T0 * dIdsa_dVd + Idsa * (one - dVdseff_dVd - T9 * dVADITS_dVd) * invVADITS;
        dIdsa_dVb = T0 * dIdsa_dVb - Idsa * dVdseff_dVb * invVADITS;
        Idsa = Idsa * T0;
    }

    /* Add CLM */
    {
        vad T0 = va_log_pos(Va / Vasat);
        vad invVa = one / Va, invVasat = one / Vasat, invCclm = one / Cclm;
        vad dT0_dVg = dVa_dVg * invVa - dVasat_dVg * invVasat;
        vad dT0_dVb = dVa_dVb * invVa - dVasat_dVb * invVasat;
        vad dT0_dVd = dVa_dVd * invVa - dVasat_dVd * invVasat;
        vad T1 = T0 * invCclm;
        vad T9 = one + T1;
        vad dT9_dVg = (dT0_dVg - T1 * dCclm_dVg) * invCclm;
        vad dT9_dVb = (dT0_dVb - T1 * dCclm_dVb) * invCclm;
        vad dT9_dVd = (dT0_dVd - T1 * dCclm_dVd) * invCclm;
        dIdsa_dVg = dIdsa_dVg * T9 + Idsa * dT9_dVg;
        dIdsa_dVb = dIdsa_dVb * T9 + Idsa * dT9_dVb;
        dIdsa_dVd = dIdsa_dVd * T9 + Idsa * dT9_dVd;
        Idsa = Idsa * T9;
    }

    /* Substrate current Isub */
    vad Isub = z0, Gbg = z0, Gbd = z0, Gbb = z0;
    if (pp->f_isub) {
        vad thr = va_set1(pp->beta0) * va_set1(1.0 / B4V_EXP_THRESHOLD);
        vam m_exp = (pp->f_isub ? va_mask_all() : va_mask_zero()) & va_gt(diffVds, thr);
        vad T3_lo = va_set1(pp->alpha_ov_leff) * min_exp;
        vad T1 = T3_lo * diffVds;
        vad dT1_dVg = -T3_lo * dVdseff_dVg;
        vad dT1_dVd = T3_lo * (one - dVdseff_dVd);
        vad dT1_dVb = -T3_lo * dVdseff_dVb;
        if (va_any(m_exp)) {
            vad diff_safe = va_max(diffVds, va_set1(1.0e-30));
            vad T0 = -va_set1(pp->beta0) / diff_safe;
            vad T1e = va_set1(pp->alpha_ov_leff) * diff_safe * va_exp_clamped(T0, -B4V_EXP_THRESHOLD, 0.0);
            vad T3e = T1e / diff_safe * (T0 - one);
            T1      = va_sel(m_exp, T1e, T1);
            dT1_dVg = va_sel(m_exp, T3e * dVdseff_dVg, dT1_dVg);
            dT1_dVd = va_sel(m_exp, T3e * (dVdseff_dVd - one), dT1_dVd);
            dT1_dVb = va_sel(m_exp, T3e * dVdseff_dVb, dT1_dVb);
        }
        vad T4 = Idsa * Vdseff;
        vad gbg0 = T1 * (dIdsa_dVg * Vdseff + Idsa * dVdseff_dVg) + T4 * dT1_dVg;
        vad gbd0 = T1 * (dIdsa_dVd * Vdseff + Idsa * dVdseff_dVd) + T4 * dT1_dVd;
        vad gbb0 = T1 * (dIdsa_dVb * Vdseff + Idsa * dVdseff_dVb) + T4 * dT1_dVb;
        gbd0 = gbd0 + gbg0 * dVgsteff_dVd;
        gbb0 = (gbb0 + gbg0 * dVgsteff_dVb) * dVbseff_dVb;
        gbg0 = gbg0 * dVgsteff_dVg;
        Isub = va_sel((pp->f_isub ? va_mask_all() : va_mask_zero()), T1 * T4, z0);
        Gbg  = va_sel((pp->f_isub ? va_mask_all() : va_mask_zero()), gbg0, z0);
        Gbd  = va_sel((pp->f_isub ? va_mask_all() : va_mask_zero()), gbd0, z0);
        Gbb  = va_sel((pp->f_isub ? va_mask_all() : va_mask_zero()), gbb0, z0);
    }

    /* Add SCBE & assemble Ids, Gm, Gds, Gmb */
    vad Ids, Gm, Gds, Gmb, cdrain;
    {
        vad invVASCBE = one / VASCBE;
        vad T9 = diffVds * invVASCBE;
        vad T0 = one + T9;
        Ids = Idsa * T0;
        vad gm0  = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVASCBE_dVg) * invVASCBE;
        vad gds0 = T0 * dIdsa_dVd + Idsa * (one - dVdseff_dVd - T9 * dVASCBE_dVd) * invVASCBE;
        vad gmb0 = T0 * dIdsa_dVb - Idsa * (dVdseff_dVb + T9 * dVASCBE_dVb) * invVASCBE;

        vad tmp1 = gds0 + gm0 * dVgsteff_dVd;
        vad tmp2 = gmb0 + gm0 * dVgsteff_dVb;
        Gm  = (Ids * dVdseff_dVg + Vdseff * gm0) * dVgsteff_dVg;
        Gds = Ids * (dVdseff_dVd + dVdseff_dVg * dVgsteff_dVd) + Vdseff * tmp1;
        Gmb = (Ids * (dVdseff_dVb + dVdseff_dVg * dVgsteff_dVb) + Vdseff * tmp2) * dVbseff_dVb;
        cdrain = Ids * Vdseff;
    }

    /* Source End Velocity Limit (vtl > 0) */
    if (pp->f_vtl) {
        vad T12 = va_set1(pp->invLeff) / CoxeffWovL;
        vad T11 = T12 / Vgsteff;
        vad T10 = -T11 / Vgsteff;
        vad vs = cdrain * T11;
        vad dvs_dVg = Gm * T11 + cdrain * T10 * dVgsteff_dVg;
        vad dvs_dVd = Gds * T11 + cdrain * T10 * dVgsteff_dVd;
        vad dvs_dVb = Gmb * T11 + cdrain * T10 * dVgsteff_dVb;
        vad T1 = vs * va_set1(pp->inv_vtl_tfactor);
        vam m_act = (pp->f_vtl ? va_mask_all() : va_mask_zero()) & va_gt(T1, z0);
        if (va_any(m_act)) {
            vad s2 = T1 * T1;
            vad T2 = one + s2 * s2 * s2;
            vad vs_safe = va_sel(m_act, vs, one);
            vad T3 = (T2 - one) * va_set1(6.0) / vs_safe;
            /* T2^(-1/6) = exp((-1/6) * log(T2)) */
            vad Fsevl = va_exp(va_set1(-1.0 / 6.0) * va_log_pos(T2));
            vad T4 = va_set1(-1.0 / 6.0) * Fsevl / T2;
            vad dF_g = T4 * T3 * dvs_dVg;
            vad dF_d = T4 * T3 * dvs_dVd;
            vad dF_b = T4 * T3 * dvs_dVb;
            Gm     = va_sel(m_act, Gm * Fsevl + cdrain * dF_g, Gm);
            Gmb    = va_sel(m_act, Gmb * Fsevl + cdrain * dF_b, Gmb);
            Gds    = va_sel(m_act, Gds * Fsevl + cdrain * dF_d, Gds);
            cdrain = va_sel(m_act, cdrain * Fsevl, cdrain);
        }
    }

    /* ===================================================================== */
    /* 8. GIDL & GISL                                                        */
    /* ===================================================================== */
    vad Igidl = z0, Ggidld = z0, Ggidlg = z0, Ggidlb = z0;
    if (pp->f_gidl_en) {
        vad T1 = (vds - vgs_eff - va_set1(pp->egidl)) * va_set1(pp->inv_3toxe);
        vam m_act = (pp->f_gidl_en ? va_mask_all() : va_mask_zero()) & va_gt(T1, z0) & va_le(vbd, z0);
        if (va_any(m_act)) {
            vad T1s = va_max(T1, va_set1(1.0e-30));
            vad dT1_dVd = va_set1(pp->inv_3toxe);
            vad dT1_dVg = -dvgs_eff_dvg * dT1_dVd;
            vad T2 = va_set1(pp->bgidl) / T1s;
            vam m_e = m_act & va_lt(T2, va_set1(100.0));
            vad ig_lo = va_set1(pp->agidl_weffCJ) * va_set1(3.720075976e-44);
            vad gd_l = ig_lo * dT1_dVd;
            vad gg_l = ig_lo * dT1_dVg;
            vad ig = ig_lo * T1s;
            vad gd = gd_l, gg = gg_l;
            if (va_any(m_e)) {
                vad ige = va_set1(pp->agidl_weffCJ) * T1s * va_exp_clamped(-T2, -100.0, 0.0);
                vad T3 = ige * (one + T2) / T1s;
                ig = va_sel(m_e, ige, ig);
                gd = va_sel(m_e, T3 * dT1_dVd, gd);
                gg = va_sel(m_e, T3 * dT1_dVg, gg);
            }
            vad T4 = vbd * vbd;
            vad T5 = -vbd * T4;
            vad T6 = va_set1(pp->cgidl) + T5;
            vad invT6 = one / T6;
            vad T7 = T5 * invT6;
            vad T8 = va_set1(3.0) * va_set1(pp->cgidl) * T4 * invT6 * invT6;
            Ggidld = va_sel(m_act, gd * T7 + ig * T8, z0);
            Ggidlg = va_sel(m_act, gg * T7, z0);
            Ggidlb = va_sel(m_act, -ig * T8, z0);
            Igidl  = va_sel(m_act, ig * T7, z0);
        }
    }

    vad Igisl = z0, Ggisls = z0, Ggislg = z0, Ggislb = z0;
    if (pp->f_gisl_en) {
        vad T1 = (-vds - vgd_eff - va_set1(pp->egisl)) * va_set1(pp->inv_3toxe);
        vam m_act = (pp->f_gisl_en ? va_mask_all() : va_mask_zero()) & va_gt(T1, z0) & va_le(vbs, z0);
        if (va_any(m_act)) {
            vad T1s = va_max(T1, va_set1(1.0e-30));
            vad dT1_dVd = va_set1(pp->inv_3toxe);
            vad dT1_dVg = -dvgd_eff_dvg * dT1_dVd;
            vad T2 = va_set1(pp->bgisl) / T1s;
            vam m_e = m_act & va_lt(T2, va_set1(100.0));
            vad ig_lo = va_set1(pp->agisl_weffCJ) * va_set1(3.720075976e-44);
            vad gs_l = ig_lo * dT1_dVd;
            vad gg_l = ig_lo * dT1_dVg;
            vad ig = ig_lo * T1s;
            vad gs = gs_l, gg = gg_l;
            if (va_any(m_e)) {
                vad ige = va_set1(pp->agisl_weffCJ) * T1s * va_exp_clamped(-T2, -100.0, 0.0);
                vad T3 = ige * (one + T2) / T1s;
                ig = va_sel(m_e, ige, ig);
                gs = va_sel(m_e, T3 * dT1_dVd, gs);
                gg = va_sel(m_e, T3 * dT1_dVg, gg);
            }
            vad T4 = vbs * vbs;
            vad T5 = -vbs * T4;
            vad T6 = va_set1(pp->cgisl) + T5;
            vad invT6 = one / T6;
            vad T7 = T5 * invT6;
            vad T8 = va_set1(3.0) * va_set1(pp->cgisl) * T4 * invT6 * invT6;
            Ggisls = va_sel(m_act, gs * T7 + ig * T8, z0);
            Ggislg = va_sel(m_act, gg * T7, z0);
            Ggislb = va_sel(m_act, -ig * T8, z0);
            Igisl  = va_sel(m_act, ig * T7, z0);
        }
    }

    /* Scale DC currents/conductances by nf */
    if (pp->f_nf_ne1) {
        cdrain = cdrain * va_set1(pp->nf);
        Gds = Gds * va_set1(pp->nf);
        Gm  = Gm  * va_set1(pp->nf);
        Gmb = Gmb * va_set1(pp->nf);
        Gbb = Gbb * va_set1(pp->nf);
        Gbg = Gbg * va_set1(pp->nf);
        Gbd = Gbd * va_set1(pp->nf);
        Isub = Isub * va_set1(pp->nf);
        Igidl = Igidl * va_set1(pp->nf); Ggidld = Ggidld * va_set1(pp->nf); Ggidlg = Ggidlg * va_set1(pp->nf); Ggidlb = Ggidlb * va_set1(pp->nf);
        Igisl = Igisl * va_set1(pp->nf); Ggisls = Ggisls * va_set1(pp->nf); Ggislg = Ggislg * va_set1(pp->nf); Ggislb = Ggislb * va_set1(pp->nf);
    }

    /* ===================================================================== */
    /* 1. Source/Drain Junction Diodes (dioMod == 1)                         */
    /* ===================================================================== */
    vad gbs, cbs, gbd, cbd;
    {
        vad T2 = vbs * va_set1(pp->invNvtms);
        vam m_low = va_lt(T2, nexp_th);
        vam m_mid = (~m_low) & va_le(vbs, va_jct_gather(jpts, vjsmFwd)) & va_jct_mask(jpts, B4A_JF_SRC_SAT);
        vad evbs = z0;
        if (va_any(m_mid))
            evbs = va_exp_clamped(T2, -B4V_EXP_THRESHOLD, 100.0);
        vad g_low = gmin;
        vad c_low = va_jct_gather(jpts, SourceSatCurrent) * (min_exp - one) + gmin * vbs;
        vad g_mid = va_jct_gather(jpts, SourceSatCurrent) * evbs * va_set1(pp->invNvtms) + gmin;
        vad c_mid = va_jct_gather(jpts, SourceSatCurrent) * (evbs - one) + gmin * vbs;
        vad T0h = va_jct_gather(jpts, IVjsmFwd) * va_set1(pp->invNvtms);
        vad g_hi = T0h + gmin;
        vad c_hi = va_jct_gather(jpts, IVjsmFwd) - va_jct_gather(jpts, SourceSatCurrent) + T0h * (vbs - va_jct_gather(jpts, vjsmFwd)) + gmin * vbs;
        gbs = va_sel(m_low, g_low, va_sel(m_mid, g_mid, g_hi));
        cbs = va_sel(m_low, c_low, va_sel(m_mid, c_mid, c_hi));
        gbs = va_sel(va_jct_mask(jpts, B4A_JF_SRC_SAT), gbs, gmin);
        cbs = va_sel(va_jct_mask(jpts, B4A_JF_SRC_SAT), cbs, gmin * vbs);
    }
    {
        vad T2 = vbd * va_set1(pp->invNvtmd);
        vam m_low = va_lt(T2, nexp_th);
        vam m_mid = (~m_low) & va_le(vbd, va_jct_gather(jpts, vjdmFwd)) & va_jct_mask(jpts, B4A_JF_DRN_SAT);
        vad evbd = z0;
        if (va_any(m_mid))
            evbd = va_exp_clamped(T2, -B4V_EXP_THRESHOLD, 100.0);
        vad g_low = gmin;
        vad c_low = va_jct_gather(jpts, DrainSatCurrent) * (min_exp - one) + gmin * vbd;
        vad g_mid = va_jct_gather(jpts, DrainSatCurrent) * evbd * va_set1(pp->invNvtmd) + gmin;
        vad c_mid = va_jct_gather(jpts, DrainSatCurrent) * (evbd - one) + gmin * vbd;
        vad T0h = va_jct_gather(jpts, IVjdmFwd) * va_set1(pp->invNvtmd);
        vad g_hi = T0h + gmin;
        vad c_hi = va_jct_gather(jpts, IVjdmFwd) - va_jct_gather(jpts, DrainSatCurrent) + T0h * (vbd - va_jct_gather(jpts, vjdmFwd)) + gmin * vbd;
        gbd = va_sel(m_low, g_low, va_sel(m_mid, g_mid, g_hi));
        cbd = va_sel(m_low, c_low, va_sel(m_mid, c_mid, c_hi));
        gbd = va_sel(va_jct_mask(jpts, B4A_JF_DRN_SAT), gbd, gmin);
        cbd = va_sel(va_jct_mask(jpts, B4A_JF_DRN_SAT), cbd, gmin * vbd);
    }
    if (pp->has_trap_tunnel) {
        if ((jct_any_flags & B4A_JF_TT_S))
            va_trap_tunnel_term(va_jct_mask(jpts, B4A_JF_TT_S), vbs, va_set1(pp->vtss), va_set1(pp->invNvtmrs_s), va_jct_gather(jpts, SjctRevSat), &gbs, &cbs);
        if ((jct_any_flags & B4A_JF_TT_D))
            va_trap_tunnel_term(va_jct_mask(jpts, B4A_JF_TT_D), vbd, va_set1(pp->vtsd), va_set1(pp->invNvtmrs_d), va_jct_gather(jpts, DjctRevSat), &gbd, &cbd);
        if ((jct_any_flags & B4A_JF_TT_SSW))
            va_trap_tunnel_term(va_jct_mask(jpts, B4A_JF_TT_SSW), vbs, va_set1(pp->vtssws), va_set1(pp->invNvtmrssw_s), va_jct_gather(jpts, SswRevSat), &gbs, &cbs);
        if ((jct_any_flags & B4A_JF_TT_DSW))
            va_trap_tunnel_term(va_jct_mask(jpts, B4A_JF_TT_DSW), vbd, va_set1(pp->vtsswd), va_set1(pp->invNvtmrssw_d), va_jct_gather(jpts, DswRevSat), &gbd, &cbd);
        if ((jct_any_flags & B4A_JF_TT_SSWG))
            va_trap_tunnel_term(va_jct_mask(jpts, B4A_JF_TT_SSWG), vbs, va_set1(pp->vtsswgs), va_set1(pp->invNvtmrsswg_s), va_jct_gather(jpts, SswgRevSat), &gbs, &cbs);
        if ((jct_any_flags & B4A_JF_TT_DSWG))
            va_trap_tunnel_term(va_jct_mask(jpts, B4A_JF_TT_DSWG), vbd, va_set1(pp->vtsswgd), va_set1(pp->invNvtmrsswg_d), va_jct_gather(jpts, DswgRevSat), &gbd, &cbd);
    }

    /* ===================================================================== */
    {
    vad ceqjs = (cbs - gbs * vbs) * va_set1(pp->type);
    vad ceqjd = (cbd - gbd * vbd) * va_set1(pp->type);

    /* Forward vs Reverse DC stamping terms */
    vad Gm_st   = va_sel(m_fwd, Gm, -Gm);
    vad Gmbs_st = va_sel(m_fwd, Gmb, -Gmb);
    vad FwdSum  = va_sel(m_fwd, Gm + Gmb, z0);
    vad RevSum  = va_sel(m_fwd, z0, -(Gm_st + Gmbs_st)); /* == Gm + Gmb when rev */

    vad ceqdrn_f = va_set1(pp->type) * (cdrain - Gds * vds - Gm * vgs - Gmb * vbs);
    vad ceqdrn_r = -va_set1(pp->type) * (cdrain + Gds * vds - Gm * vgd - Gmb * vbd);
    vad ceqdrn = va_sel(m_fwd, ceqdrn_f, ceqdrn_r);

    vad ceqbd_f = va_set1(pp->type) * (Isub + Igidl - (Gbd + Ggidld) * vds - (Gbg + Ggidlg) * vgs - (Gbb + Ggidlb) * vbs);
    vad ceqbs_f = va_set1(pp->type) * (Igisl + Ggisls * vds - Ggislg * vgd - Ggislb * vbd);
    vad ceqbd_r = va_set1(pp->type) * (Igidl - Ggidld * vds - Ggidlg * vgs - Ggidlb * vbs);
    vad ceqbs_r = va_set1(pp->type) * (Isub + Igisl + (Gbd + Ggisls) * vds - (Gbg + Ggislg) * vgd - (Gbb + Ggislb) * vbd);
    vad ceqbd = va_sel(m_fwd, ceqbd_f, ceqbd_r);
    vad ceqbs = va_sel(m_fwd, ceqbs_f, ceqbs_r);

    vad gb_sum = Gbd + Gbg + Gbb;
    vad gbbdp  = va_sel(m_fwd, -Gbd, gb_sum);
    vad gbbsp  = va_sel(m_fwd, gb_sum, -Gbd);
    vad gbdpg  = va_sel(m_fwd, Gbg, z0);
    vad gbdpdp = va_sel(m_fwd, Gbd, z0);
    vad gbdpb  = va_sel(m_fwd, Gbb, z0);
    vad gbdpsp = va_sel(m_fwd, -gb_sum, z0);
    vad gbspg  = va_sel(m_fwd, z0, Gbg);
    vad gbspdp = va_sel(m_fwd, z0, -gb_sum);
    vad gbspb  = va_sel(m_fwd, z0, Gbb);
    vad gbspsp = va_sel(m_fwd, z0, Gbd);

    vad ggidl_sum = Ggidlg + Ggidld + Ggidlb;
    vad ggisl_sum = Ggislg + Ggisls + Ggislb;

    

    vad vm = m_v;
    B4A_SCATTER(l2pts, mdc44, vm * (drainCond_v + Gds + gbd + RevSum + gbdpdp + Ggidld));
    B4A_SCATTER(l2pts, mdc46, vm * (Gm_st + gbdpg + Ggidlg));
    B4A_SCATTER(l2pts, mdc47, vm * (Gds + FwdSum - gbdpsp + ggidl_sum));
    B4A_SCATTER(l2pts, mdc48, vm * (gbd - Gmbs_st - gbdpb - Ggidlb));
    B4A_SCATTER(l2pts, mdc51, vm * (Gds + RevSum - gbspdp + ggisl_sum));
    B4A_SCATTER(l2pts, mdc52, vm * (-Gm_st + gbspg + Ggislg));
    B4A_SCATTER(l2pts, mdc53, vm * (sourceCond_v + Gds + gbs + FwdSum + gbspsp + Ggisls));
    B4A_SCATTER(l2pts, mdc55, vm * (gbs + Gmbs_st - gbspb - Ggislb));
    B4A_SCATTER(l2pts, mdc58, vm * (-gbd + gbbdp - Ggidld + ggisl_sum));
    B4A_SCATTER(l2pts, mdc59, vm * (-Gbg - Ggidlg - Ggislg));
    B4A_SCATTER(l2pts, mdc60, vm * (-gbs + gbbsp + ggidl_sum - Ggisls));
    B4A_SCATTER(l2pts, mdc61, vm * (gbd + gbs - Gbb - Ggidlb - Ggislb));
    B4A_SCATTER(l2pts, mdc_rhsd, vm * (ceqjd - ceqbd - ceqdrn));
    B4A_SCATTER(l2pts, mdc_rhsb, vm * (ceqbd + ceqbs - ceqjd - ceqjs));
    B4A_SCATTER(l2pts, mdc_rhss, vm * (ceqdrn - ceqbs + ceqjs));

    {
        vad Ggidls = -ggidl_sum;
        vad Ggisld = -ggisl_sum;
        vad id_f = cdrain + Isub - cbd + Igidl;
        vad id_r = cdrain + cbd - Igidl;
        B4A_SCATTER(cpts, Idtot, va_sel(m_fwd, id_f, id_r));
        B4A_SCATTER(cpts, cd_g,  va_sel(m_fwd, Gm + Gbg + Ggidlg, Gm - Ggidlg));
        B4A_SCATTER(cpts, cd_b,  va_sel(m_fwd, Gmb + Gbb + Ggidlb - gbd, gbd + Gmb - Ggidlb));
        B4A_SCATTER(cpts, cd_d,  va_sel(m_fwd, Gds + Gbd + Ggidld + gbd, -(gbd + Gmb + Gm + Gds + Ggidls)));
        B4A_SCATTER(cpts, Ibtot, cbs + cbd - Igidl - Igisl - Isub);
        B4A_SCATTER(cpts, cb_g,  -(Gbg + Ggidlg + Ggislg));
        B4A_SCATTER(cpts, cb_b,  gbd + gbs - (Gbb + Ggidlb + Ggislb));
        vad cbd_f = -gbd - (Gbd + Ggidld) + Ggislg + Ggislb + Ggisls;
        vad cbd_r = -gbd + (Gbb + Ggislb) + (Gbg + Ggislg) + (Gbd + Ggisld - Ggidls);
        B4A_SCATTER(cpts, cb_d,  va_sel(m_fwd, cbd_f, cbd_r));
    }

    if (need_aos && out) {
        va_store(out->von, p->Vth);
        for (int l = 0; l < nact; l++)
            out->mode[l] = va_mask_bit(m_fwd, l) ? 1 : -1;
        va_store(out->cd, cdrain);
        va_store(out->gds, Gds);
        va_store(out->gm, Gm);
        va_store(out->gmbs, Gmb);
        va_store(out->gbd, gbd);
        va_store(out->gbs, gbs);
        va_store(out->cbs, cbs);
        va_store(out->cbd, cbd);
        va_store(out->csub, Isub);
        va_store(out->gbds, Gbd);
        va_store(out->gbgs, Gbg);
        va_store(out->gbbs, Gbb);
        va_store(out->Igidl, Igidl);
        va_store(out->ggidld, Ggidld);
        va_store(out->ggidlg, Ggidlg);
        va_store(out->ggidlb, Ggidlb);
        va_store(out->Igisl, Igisl);
        va_store(out->ggisls, Ggisls);
        va_store(out->ggislg, Ggislg);
        va_store(out->ggislb, Ggislb);
    }
    }
}

__attribute__((noinline))
static void
bsim4_apple_p3_cv_jct(
    const BSIM4ApplePParam * restrict pp,
    const BSIM4ApplePipe * restrict p,
    const BSIM4AppleJct * const jpts[B4A_W],
    BSIM4AppleCheck * const cpts[B4A_W],
    BSIM4AppleStamp * const spts[B4A_W],
    BSIM4AppleL2 * const l2pts[B4A_W],
    int nact,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2,
    int need_aos)
{
    const vad z0 = va_set1(0.0);
    const vad one = va_set1(1.0);
    const vad half = va_set1(0.5);
    const vad two = va_set1(2.0);
    const vad max_exp = va_set1(B4V_MAX_EXP);
    const vad min_exp = va_set1(B4V_MIN_EXP);
    const vad exp_th = va_set1(B4V_EXP_THRESHOLD);
    const vad nexp_th = va_set1(-B4V_EXP_THRESHOLD);
    const vad epssi = va_set1(B4V_EPSSI);
    (void)max_exp; (void)min_exp;

    uint16_t jct_any_flags = 0;
    for (int l = 0; l < B4A_W; l++)
        jct_any_flags |= jpts[l]->flags;

    const vad vds = p->vds;
    const vad vgs = p->vgs;
    const vad vbs = p->vbs;
    const vad vbd = p->vbd;
    const vad vgd = p->vgd;
    const vad vgb = p->vgb;
    const vad Vds = p->Vds;
    const vad Vgs_eff = p->Vgs_eff;
    const vad dVgs_eff_dVg = p->dVgs_eff_dVg;
    const vad Vbseff = p->Vbseff;
    const vad dVbseff_dVb = p->dVbseff_dVb;
    const vad Phis = p->Phis;
    const vad dVth_dVd = p->dVth_dVd;
    const vad dVth_dVb = p->dVth_dVb;
    const vad n = p->n;
    const vad dn_dVd = p->dn_dVd;
    const vad dn_dVb = p->dn_dVb;
    const vad Vgst = p->Vgst;
    const vad Abulk0 = p->Abulk0;
    const vad dAbulk0_dVb = p->dAbulk0_dVb;
    const vam m_fwd = p->m_fwd;


    /* 9. Intrinsic C-V (capMod == 2, Charge-Thickness Model)                */
    /* ===================================================================== */
    /* Note: if Vbseff < 0 -> VbseffCV = Vbseff, dVbseffCV_dVb = 1.0;
     *       else -> VbseffCV = phi - Phis = Vbseff, dVbseffCV_dVb = -(-1.0) = 1.0! */
    vad VbseffCV = Vbseff;
    vad VgsteffCV, dVgCV_dVg, dVgCV_dVd, dVgCV_dVb;
    {
        vad noff = n * va_set1(pp->noff);
        vad dnoff_dVd = va_set1(pp->noff) * dn_dVd;
        vad dnoff_dVb = va_set1(pp->noff) * dn_dVb;
        vad T0 = va_set1(pp->vtm) * noff;
        vad dVg_off = Vgst - va_set1(pp->voffcv);
        vad VgstNVt = dVg_off / T0;
        vam m_hi = va_gt(VgstNVt, exp_th);
        vam m_lo = va_lt(VgstNVt, nexp_th);
        vam m_mid = (~m_hi) & (~m_lo);

        vad vg_lo = T0 * va_set1(B4V_LOG1P_MIN_EXP);
        vad inv_noff = one / noff;
        vad ratio_lo = vg_lo * inv_noff;
        VgsteffCV = va_sel(m_hi, dVg_off, vg_lo);
        dVgCV_dVg = va_sel(m_hi, dVgs_eff_dVg, z0);
        dVgCV_dVd = va_sel(m_hi, -dVth_dVd, ratio_lo * dnoff_dVd);
        dVgCV_dVb = va_sel(m_hi, -dVth_dVb, ratio_lo * dnoff_dVb);
        if (va_any(m_mid)) {
            vad ex = va_exp_clamped(VgstNVt, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            vad vg = T0 * va_log_pos(one + ex);
            vad dg0 = ex / (one + ex);
            VgsteffCV = va_sel(m_mid, vg, VgsteffCV);
            dVgCV_dVd = va_sel(m_mid, -dg0 * (dVth_dVd + dVg_off * inv_noff * dnoff_dVd) + vg * inv_noff * dnoff_dVd, dVgCV_dVd);
            dVgCV_dVb = va_sel(m_mid, -dg0 * (dVth_dVb + dVg_off * inv_noff * dnoff_dVb) + vg * inv_noff * dnoff_dVb, dVgCV_dVb);
            dVgCV_dVg = va_sel(m_mid, dg0 * dVgs_eff_dVg, dVgCV_dVg);
        }
    }

    vad qgate, qbulk, qdrn;
    vad cggb, cgsb, cgdb, cdgb, cdsb, cddb, cbgb, cbsb, cbdb;
    {
        vad V3 = va_set1(pp->vfbzb) - Vgs_eff + VbseffCV - va_set1(B4V_DELTA_3);
        vad f4 = va_sel((pp->f_vfbzb_le0 ? va_mask_all() : va_mask_zero()), va_set1(-4.0 * B4V_DELTA_3), va_set1(4.0 * B4V_DELTA_3));
        vad T0_fb = va_sqrt(V3 * V3 + f4 * va_set1(pp->vfbzb));
        vad T1_fb = half * (one + V3 / T0_fb);
        vad Vfbeff = va_set1(pp->vfbzb) - half * (V3 + T0_fb);
        vad dVfbeff_dVg = T1_fb * dVgs_eff_dVg;
        vad dVfbeff_dVb = -T1_fb;

        vad T0_ox = (Vgs_eff - VbseffCV - va_set1(pp->vfbzb)) * va_set1(pp->inv_tox_ctm);
        vad dT0_ox_dVg = dVgs_eff_dVg * va_set1(pp->inv_tox_ctm);
        vad dT0_ox_dVb = -va_set1(pp->inv_tox_ctm);
        vad tmp_ac = T0_ox * va_set1(pp->acde);
        vam m_ac_lo = va_le(tmp_ac, nexp_th);
        vam m_ac_hi = va_ge(tmp_ac, exp_th);
        vam m_ac_mid = (~m_ac_lo) & (~m_ac_hi);
        vad Tcen = va_set1(pp->ldeb) * va_sel(m_ac_lo, min_exp, max_exp);
        vad dTcen_dVg = z0, dTcen_dVb = z0;
        if (va_any(m_ac_mid)) {
            vad tc = va_set1(pp->ldeb) * va_exp_clamped(tmp_ac, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            vad dtc = va_set1(pp->acde) * tc;
            Tcen = va_sel(m_ac_mid, tc, Tcen);
            dTcen_dVb = va_sel(m_ac_mid, dtc * dT0_ox_dVb, z0);
            dTcen_dVg = va_sel(m_ac_mid, dtc * dT0_ox_dVg, z0);
        }

        vad V3l = va_set1(pp->ldeb) - Tcen - va_set1(pp->link_ctm);
        vad V4l = va_sqrt(V3l * V3l + va_set1(4.0) * va_set1(pp->link_ctm) * va_set1(pp->ldeb));
        Tcen = va_set1(pp->ldeb) - half * (V3l + V4l);
        vad T1l = half * (one + V3l / V4l);
        dTcen_dVg = dTcen_dVg * T1l;
        dTcen_dVb = dTcen_dVb * T1l;

        vad Ccen = epssi / Tcen;
        vad T2c = va_set1(pp->coxp) / (va_set1(pp->coxp) + Ccen);
        vad Coxeff = T2c * Ccen;
        vad dCox_base = T2c * T2c * (-Ccen / Tcen);
        vad dCoxeff_dVg = dCox_base * dTcen_dVg;
        vad dCoxeff_dVb = dCox_base * dTcen_dVb;
        vad CoxWLcen = va_set1(pp->CoxWL_ov_coxe) * Coxeff;

        vad Qac0 = CoxWLcen * (Vfbeff - va_set1(pp->vfbzb));
        vad QovCox_ac = Qac0 / Coxeff;
        vad dQac0_dVg = CoxWLcen * dVfbeff_dVg + QovCox_ac * dCoxeff_dVg;
        vad dQac0_dVb = CoxWLcen * dVfbeff_dVb + QovCox_ac * dCoxeff_dVb;

        vad T0s = half * va_set1(pp->k1ox);
        vad T3s = Vgs_eff - Vfbeff - VbseffCV - VgsteffCV;
        vam m_t3neg = va_lt(T3s, z0);
        vad T1_pos = va_sqrt(va_max(T0s * T0s + T3s, va_set1(1.0e-300)));
        vad T1_neg = T0s + T3s / va_sel((pp->f_k1ox_zero ? va_mask_all() : va_mask_zero()), one, va_set1(pp->k1ox));
        vad T1s = va_sel((pp->f_k1ox_zero ? va_mask_all() : va_mask_zero()), z0, va_sel(m_t3neg, T1_neg, T1_pos));
        vad T2s = va_sel((pp->f_k1ox_zero ? va_mask_all() : va_mask_zero()), z0, va_sel(m_t3neg, CoxWLcen, CoxWLcen * T0s / T1_pos));
        vad Qsub0 = CoxWLcen * va_set1(pp->k1ox) * (T1s - T0s);
        vad QovCox_sub = Qsub0 / Coxeff;
        vad dQsub0_dVg = T2s * (dVgs_eff_dVg - dVfbeff_dVg - dVgCV_dVg) + QovCox_sub * dCoxeff_dVg;
        vad dQsub0_dVd = -T2s * dVgCV_dVd;
        vad dQsub0_dVb = -T2s * (dVfbeff_dVb + one + dVgCV_dVb) + QovCox_sub * dCoxeff_dVb;

        /* DeltaPhi & VgDP */
        vad Den_dp = va_sel((pp->f_k1ox_le0 ? va_mask_all() : va_mask_zero()), va_set1(0.25) * va_set1(pp->moin) * va_set1(pp->vtm),
                            va_set1(pp->moin) * va_set1(pp->vtm) * va_set1(pp->k1ox) * va_set1(pp->k1ox));
        vad T0_dp = va_sel((pp->f_k1ox_le0 ? va_mask_all() : va_mask_zero()), half * va_set1(pp->sqrtPhi), va_set1(pp->k1ox) * va_set1(pp->sqrtPhi));
        vad T1_dp = two * T0_dp + VgsteffCV;
        vad DeltaPhi = va_set1(pp->vtm) * va_log_pos(one + T1_dp * VgsteffCV / Den_dp);
        vad dDeltaPhi_dVg = two * va_set1(pp->vtm) * (T1_dp - T0_dp) / (Den_dp + T1_dp * VgsteffCV);

        vad T0_vg = VgsteffCV - DeltaPhi - va_set1(0.001);
        vad dT0_vg = one - dDeltaPhi_dVg;
        vad T1_vg = va_sqrt(T0_vg * T0_vg + VgsteffCV * va_set1(0.004));
        vad VgDP = half * (T0_vg + T1_vg);
        vad dVgDP_dVg = half * (dT0_vg + (T0_vg * dT0_vg + va_set1(0.002)) / T1_vg);

        /* Inversion Tcen & Coxeff */
        vad T0_i = (VgsteffCV + va_set1(pp->vtfbphi2)) * va_set1(pp->inv_2e8_toxp);
        vad tmp_i = va_exp(va_set1(pp->bdos0p7) * va_log_pos(T0_i));
        vad T1_i = one + tmp_i;
        vad invT1_i = one / T1_i;
        vad T2_i = va_set1(pp->bdos0p7) * tmp_i * va_set1(pp->inv_2e8_toxp) / T0_i;
        vad Tcen_i = va_set1(pp->ados1p9) * invT1_i;
        vad dTcen_i0 = -Tcen_i * T2_i * invT1_i;
        vad dTcen_id = dTcen_i0 * dVgCV_dVd;
        vad dTcen_ib = dTcen_i0 * dVgCV_dVb;
        vad dTcen_ig = dTcen_i0 * dVgCV_dVg;

        vad Ccen_i = epssi / Tcen_i;
        vad T0_ci = va_set1(pp->coxp) / (va_set1(pp->coxp) + Ccen_i);
        vad Coxeff_i = T0_ci * Ccen_i;
        vad dCox_i0 = T0_ci * T0_ci * (-Ccen_i / Tcen_i);
        vad dCoxeff_id = dCox_i0 * dTcen_id;
        vad dCoxeff_ib = dCox_i0 * dTcen_ib;
        vad dCoxeff_ig = dCox_i0 * dTcen_ig;
        vad CoxWLcen_i = va_set1(pp->CoxWL_ov_coxe) * Coxeff_i;

        /* VdsatCV & VdseffCV */
        vad AbulkCV = Abulk0 * va_set1(pp->abulkCVfactor);
        vad dAbulkCV_dVb = va_set1(pp->abulkCVfactor) * dAbulk0_dVb;
        vad invAbCV = one / AbulkCV;
        vad VdsatCV = VgDP * invAbCV;

        vad T0_v = VdsatCV - Vds - va_set1(B4V_DELTA_4);
        vad dT0_vg_cv = dVgDP_dVg * invAbCV;
        vad dT0_vb_cv = -VdsatCV * dAbulkCV_dVb * invAbCV;
        vad T1_v = va_sqrt(T0_v * T0_v + va_set1(4.0 * B4V_DELTA_4) * VdsatCV);
        vad invT1_v = one / T1_v;
        vad dT1_base = (T0_v + va_set1(2.0 * B4V_DELTA_4)) * invT1_v;
        vad dT1_vd_cv = -T0_v * invT1_v;
        vad dT1_vb_cv = dT1_base * dT0_vb_cv;
        vad dT1_vg_cv = dT1_base * dT0_vg_cv;

        vam m_ge = va_ge(T0_v, z0);
        vad inv_diff_v = one / (T1_v - T0_v);
        vad T3_v = va_set1(2.0 * B4V_DELTA_4) * inv_diff_v;
        vad T4_v = one - T3_v;
        vad T5_v = VdsatCV * T3_v * inv_diff_v;

        vad VdseffCV     = va_sel(m_ge, VdsatCV - half * (T0_v + T1_v), VdsatCV * T4_v);
        vad dVdseffCV_dVg = va_sel(m_ge, half * (dT0_vg_cv - dT1_vg_cv), dT0_vg_cv * T4_v + T5_v * (dT1_vg_cv - dT0_vg_cv));
        vad dVdseffCV_dVd = va_sel(m_ge, half * (one - dT1_vd_cv), T5_v * (dT1_vd_cv + one));
        vad vb_fac = pp->is_v6 ? (T4_v - T5_v) : (one - T5_v);
        vad dVdseffCV_dVb = va_sel(m_ge, half * (dT0_vb_cv - dT1_vb_cv), dT0_vb_cv * vb_fac + T5_v * dT1_vb_cv);

        vam m_vds0 = va_eq(Vds, z0);
        VdseffCV      = va_sel(m_vds0, z0, VdseffCV);
        dVdseffCV_dVg = va_sel(m_vds0, z0, dVdseffCV_dVg);
        dVdseffCV_dVb = va_sel(m_vds0, z0, dVdseffCV_dVb);

        /* Channel charges qgate, qbulk, qsrc */
        vad T0_q = AbulkCV * VdseffCV;
        vad T1_q = VgDP;
        vad T2_q = va_set1(12.0) * (T1_q - half * T0_q + va_set1(1.0e-20));
        vad invT2_q = one / T2_q;
        vad T3_q = T0_q * invT2_q;
        vad T4_q = one - va_set1(12.0) * T3_q * T3_q;
        vad T5_q = AbulkCV * (va_set1(6.0) * T0_q * (va_set1(4.0) * T1_q - T0_q) * (invT2_q * invT2_q) - half);
        vad T6_q = T5_q * VdseffCV * invAbCV;

        vad invCox_i = one / Coxeff_i;
        qgate = CoxWLcen_i * (T1_q - T0_q * (half - T3_q));
        vad QovCox_g = qgate * invCox_i;
        vad Cgg1_0 = CoxWLcen_i * (T4_q * dVgDP_dVg + T5_q * dVdseffCV_dVg);
        vad Cgd1 = CoxWLcen_i * T5_q * dVdseffCV_dVd + Cgg1_0 * dVgCV_dVd + QovCox_g * dCoxeff_id;
        vad Cgb1 = CoxWLcen_i * (T5_q * dVdseffCV_dVb + T6_q * dAbulkCV_dVb) + Cgg1_0 * dVgCV_dVb + QovCox_g * dCoxeff_ib;
        vad Cgg1 = Cgg1_0 * dVgCV_dVg + QovCox_g * dCoxeff_ig;

        vad T7_q = one - AbulkCV;
        vad T8_q = T2_q * T2_q;
        vad T9_q = va_set1(12.0) * T7_q * T0_q * T0_q / (T8_q * AbulkCV);
        vad T10_q = T9_q * dVgDP_dVg;
        vad T11_q = -T7_q * T5_q * invAbCV;
        vad T12_q = -(T9_q * T1_q * invAbCV + VdseffCV * (half - T3_q));

        qbulk = CoxWLcen_i * T7_q * (half * VdseffCV - T0_q * VdseffCV * invT2_q);
        vad QovCox_b = qbulk * invCox_i;
        vad Cbg1_0 = CoxWLcen_i * (T10_q + T11_q * dVdseffCV_dVg);
        vad Cbd1 = CoxWLcen_i * T11_q * dVdseffCV_dVd + Cbg1_0 * dVgCV_dVd + QovCox_b * dCoxeff_id;
        vad Cbb1 = CoxWLcen_i * (T11_q * dVdseffCV_dVb + T12_q * dAbulkCV_dVb) + Cbg1_0 * dVgCV_dVb + QovCox_b * dCoxeff_ib;
        vad Cbg1 = Cbg1_0 * dVgCV_dVg + QovCox_b * dCoxeff_ig;

        vad qsrc, Csg, Csd, Csb;
        if (pp->xpart_mode == 0) {
            /* All 8 lanes use 40/60 partition (standard GF180 xpart=0) */
            vad T2p = T2_q * va_set1(1.0 / 12.0);
            vad invT2p = one / T2p;
            vad T3p = half * CoxWLcen_i * (invT2p * invT2p);
            vad T4p = T1_q * (va_set1(2.0 / 3.0) * T0_q * T0_q + T1_q * (T1_q - va_set1(4.0 / 3.0) * T0_q))
                    - va_set1(2.0 / 15.0) * T0_q * T0_q * T0_q;
            qsrc = -T3p * T4p;
            vad QovCox_s = qsrc * invCox_i;
            vad T8p = va_set1(4.0 / 3.0) * T1_q * (T1_q - T0_q) + va_set1(0.4) * T0_q * T0_q;
            vad T5p = -two * qsrc * invT2p - T3p * (T1_q * (va_set1(3.0) * T1_q - va_set1(8.0 / 3.0) * T0_q)
                    + va_set1(2.0 / 3.0) * T0_q * T0_q);
            vad T6p = AbulkCV * (qsrc * invT2p + T3p * T8p);
            vad T7p = T6p * VdseffCV * invAbCV;
            vad Csg0 = T5p * dVgDP_dVg + T6p * dVdseffCV_dVg;
            Csd = Csg0 * dVgCV_dVd + T6p * dVdseffCV_dVd + QovCox_s * dCoxeff_id;
            Csb = Csg0 * dVgCV_dVb + T6p * dVdseffCV_dVb + T7p * dAbulkCV_dVb + QovCox_s * dCoxeff_ib;
            Csg = Csg0 * dVgCV_dVg + QovCox_s * dCoxeff_ig;
        } else if (pp->xpart_mode == 2) {
            /* All 8 lanes use 50/50 partition */
            qsrc = -half * qgate;
            Csg  = -half * Cgg1;
            Csd  = -half * Cgd1;
            Csb  = -half * Cgb1;
        } else {
            /* 0/100 or mixed: start with 50/50 then blend 0/100 and 40/60 */
            qsrc = -half * qgate;
            Csg  = -half * Cgg1;
            Csd  = -half * Cgd1;
            Csb  = -half * Cgb1;
            if (pp->f_xpart_gt05) {
                vad qs0 = -CoxWLcen_i * (half * T1_q + va_set1(0.25) * T0_q - half * T0_q * T0_q * invT2_q);
                vad Qov = qs0 * invCox_i;
                vad T2d = T2_q + T2_q;
                vad invT3d = one / (T2d * T2d);
                vad T7d = -(va_set1(0.25) - va_set1(12.0) * T0_q * (va_set1(4.0) * T1_q - T0_q) * invT3d);
                vad T4d = -(half + va_set1(24.0) * T0_q * T0_q * invT3d) * dVgDP_dVg;
                vad T5d = T7d * AbulkCV;
                vad T6d = T7d * VdseffCV;
                vad Csg0 = CoxWLcen_i * (T4d + T5d * dVdseffCV_dVg);
                qsrc = va_sel((pp->f_xpart_gt05 ? va_mask_all() : va_mask_zero()), qs0, qsrc);
                Csd  = va_sel((pp->f_xpart_gt05 ? va_mask_all() : va_mask_zero()), CoxWLcen_i * T5d * dVdseffCV_dVd + Csg0 * dVgCV_dVd + Qov * dCoxeff_id, Csd);
                Csb  = va_sel((pp->f_xpart_gt05 ? va_mask_all() : va_mask_zero()), CoxWLcen_i * (T5d * dVdseffCV_dVb + T6d * dAbulkCV_dVb) + Csg0 * dVgCV_dVb + Qov * dCoxeff_ib, Csb);
                Csg  = va_sel((pp->f_xpart_gt05 ? va_mask_all() : va_mask_zero()), Csg0 * dVgCV_dVg + Qov * dCoxeff_ig, Csg);
            }
            if (pp->f_xpart_lt05) {
                vad T2p = T2_q * va_set1(1.0 / 12.0);
                vad invT2p = one / T2p;
                vad T3p = half * CoxWLcen_i * (invT2p * invT2p);
                vad T4p = T1_q * (va_set1(2.0 / 3.0) * T0_q * T0_q + T1_q * (T1_q - va_set1(4.0 / 3.0) * T0_q))
                        - va_set1(2.0 / 15.0) * T0_q * T0_q * T0_q;
                vad qs4 = -T3p * T4p;
                vad Qov = qs4 * invCox_i;
                vad T8p = va_set1(4.0 / 3.0) * T1_q * (T1_q - T0_q) + va_set1(0.4) * T0_q * T0_q;
                vad T5p = -two * qs4 * invT2p - T3p * (T1_q * (va_set1(3.0) * T1_q - va_set1(8.0 / 3.0) * T0_q)
                        + va_set1(2.0 / 3.0) * T0_q * T0_q);
                vad T6p = AbulkCV * (qs4 * invT2p + T3p * T8p);
                vad T7p = T6p * VdseffCV * invAbCV;
                vad Csg0 = T5p * dVgDP_dVg + T6p * dVdseffCV_dVg;
                qsrc = va_sel((pp->f_xpart_lt05 ? va_mask_all() : va_mask_zero()), qs4, qsrc);
                Csd  = va_sel((pp->f_xpart_lt05 ? va_mask_all() : va_mask_zero()), Csg0 * dVgCV_dVd + T6p * dVdseffCV_dVd + Qov * dCoxeff_id, Csd);
                Csb  = va_sel((pp->f_xpart_lt05 ? va_mask_all() : va_mask_zero()), Csg0 * dVgCV_dVb + T6p * dVdseffCV_dVb + T7p * dAbulkCV_dVb + Qov * dCoxeff_ib, Csb);
                Csg  = va_sel((pp->f_xpart_lt05 ? va_mask_all() : va_mask_zero()), Csg0 * dVgCV_dVg + Qov * dCoxeff_ig, Csg);
            }
        }

        qgate = qgate + Qac0 + Qsub0 - qbulk;
        qbulk = qbulk - (Qac0 + Qsub0);
        qdrn  = -(qgate + qbulk + qsrc);

        vad Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
        vad Cbd = Cbd1 - dQsub0_dVd;
        vad Cbb = (Cbb1 - dQac0_dVb - dQsub0_dVb) * dVbseff_dVb;
        vad Cgg = Cgg1 - Cbg;
        vad Cgd = Cgd1 - Cbd;
        vad Cgb = (Cgb1 - (Cbb1 - dQac0_dVb - dQsub0_dVb)) * dVbseff_dVb;
        Csb = Csb * dVbseff_dVb;

        cggb = Cgg;
        cgsb = -(Cgg + Cgd + Cgb);
        cgdb = Cgd;
        cdgb = -(Cgg + Cbg + Csg);
        cdsb = Cgg + Cgd + Cgb + Cbg + Cbd + Cbb + Csg + Csd + Csb;
        cddb = -(Cgd + Cbd + Csd);
        cbgb = Cbg;
        cbsb = -(Cbg + Cbd + Cbb);
        cbdb = Cbd;
    }

    /* ===================================================================== */
    /* 10. Junction Depletion Capacitances (qbs, capbs, qbd, capbd)          */
    /* ===================================================================== */
    vad qbs, capbs, qbd, capbd;
    {
        /* Source-bulk */
        vam m_neg = va_lt(vbs, z0);
        vad T1_f = vbs * va_jct_gather(jpts, jctS_lin);
        qbs   = vbs * (va_jct_gather(jpts, jctS_sum) + half * T1_f);
        capbs = va_jct_gather(jpts, jctS_sum) + T1_f;
        if (va_any(m_neg)) {
            vad q_n = z0, c_n = z0;
            if ((jct_any_flags & B4A_JF_CZBS) && va_any(va_jct_mask(jpts, B4A_JF_CZBS) & m_neg)) {
                vam m = va_jct_mask(jpts, B4A_JF_CZBS) & m_neg;
                vad arg = va_sel(m, one - vbs * va_set1(pp->invPhiBS), one);
                vad sarg;
                if (!va_any((~(pp->f_mjs_half ? va_mask_all() : va_mask_zero())) & m))
                    sarg = one / va_sqrt(arg);
                else {
                    sarg = va_exp(-va_set1(pp->MJS) * va_log(arg));
                    if (pp->f_mjs_half && va_any((pp->f_mjs_half ? va_mask_all() : va_mask_zero()) & m))
                        sarg = va_sel((pp->f_mjs_half ? va_mask_all() : va_mask_zero()) & m, one / va_sqrt(arg), sarg);
                }
                q_n = va_sel(m, q_n + va_set1(pp->PhiBS) * va_jct_gather(jpts, czbs) * (one - arg * sarg) * va_set1(pp->inv1mMJS), q_n);
                c_n = va_sel(m, c_n + va_jct_gather(jpts, czbs) * sarg, c_n);
            }
            if ((jct_any_flags & B4A_JF_CZBSSW) && va_any(va_jct_mask(jpts, B4A_JF_CZBSSW) & m_neg)) {
                vam m = va_jct_mask(jpts, B4A_JF_CZBSSW) & m_neg;
                vad arg = va_sel(m, one - vbs * va_set1(pp->invPhiBSWS), one);
                vad sarg;
                if (!va_any((~(pp->f_mjsws_half ? va_mask_all() : va_mask_zero())) & m))
                    sarg = one / va_sqrt(arg);
                else {
                    sarg = va_exp(-va_set1(pp->MJSWS) * va_log(arg));
                    if (pp->f_mjsws_half && va_any((pp->f_mjsws_half ? va_mask_all() : va_mask_zero()) & m))
                        sarg = va_sel((pp->f_mjsws_half ? va_mask_all() : va_mask_zero()) & m, one / va_sqrt(arg), sarg);
                }
                q_n = va_sel(m, q_n + va_set1(pp->PhiBSWS) * va_jct_gather(jpts, czbssw) * (one - arg * sarg) * va_set1(pp->inv1mMJSWS), q_n);
                c_n = va_sel(m, c_n + va_jct_gather(jpts, czbssw) * sarg, c_n);
            }
            if ((jct_any_flags & B4A_JF_CZBSSWG) && va_any(va_jct_mask(jpts, B4A_JF_CZBSSWG) & m_neg)) {
                vam m = va_jct_mask(jpts, B4A_JF_CZBSSWG) & m_neg;
                vad arg = va_sel(m, one - vbs * va_set1(pp->invPhiBSWGS), one);
                vad sarg;
                if (!va_any((~(pp->f_mjswgs_half ? va_mask_all() : va_mask_zero())) & m))
                    sarg = one / va_sqrt(arg);
                else {
                    sarg = va_exp(-va_set1(pp->MJSWGS) * va_log(arg));
                    if (pp->f_mjswgs_half && va_any((pp->f_mjswgs_half ? va_mask_all() : va_mask_zero()) & m))
                        sarg = va_sel((pp->f_mjswgs_half ? va_mask_all() : va_mask_zero()) & m, one / va_sqrt(arg), sarg);
                }
                q_n = va_sel(m, q_n + va_set1(pp->PhiBSWGS) * va_jct_gather(jpts, czbsswg) * (one - arg * sarg) * va_set1(pp->inv1mMJSWGS), q_n);
                c_n = va_sel(m, c_n + va_jct_gather(jpts, czbsswg) * sarg, c_n);
            }
            qbs   = va_sel(m_neg, q_n, qbs);
            capbs = va_sel(m_neg, c_n, capbs);
        }
    }
    {
        /* Drain-bulk */
        vam m_neg = va_lt(vbd, z0);
        vad T1_f = vbd * va_jct_gather(jpts, jctD_lin);
        qbd   = vbd * (va_jct_gather(jpts, jctD_sum) + half * T1_f);
        capbd = va_jct_gather(jpts, jctD_sum) + T1_f;
        if (va_any(m_neg)) {
            vad q_n = z0, c_n = z0;
            if ((jct_any_flags & B4A_JF_CZBD) && va_any(va_jct_mask(jpts, B4A_JF_CZBD) & m_neg)) {
                vam m = va_jct_mask(jpts, B4A_JF_CZBD) & m_neg;
                vad arg = va_sel(m, one - vbd * va_set1(pp->invPhiBD), one);
                vad sarg;
                if (!va_any((~(pp->f_mjd_half ? va_mask_all() : va_mask_zero())) & m))
                    sarg = one / va_sqrt(arg);
                else {
                    sarg = va_exp(-va_set1(pp->MJD) * va_log(arg));
                    if (pp->f_mjd_half && va_any((pp->f_mjd_half ? va_mask_all() : va_mask_zero()) & m))
                        sarg = va_sel((pp->f_mjd_half ? va_mask_all() : va_mask_zero()) & m, one / va_sqrt(arg), sarg);
                }
                q_n = va_sel(m, q_n + va_set1(pp->PhiBD) * va_jct_gather(jpts, czbd) * (one - arg * sarg) * va_set1(pp->inv1mMJD), q_n);
                c_n = va_sel(m, c_n + va_jct_gather(jpts, czbd) * sarg, c_n);
            }
            if ((jct_any_flags & B4A_JF_CZBDSW) && va_any(va_jct_mask(jpts, B4A_JF_CZBDSW) & m_neg)) {
                vam m = va_jct_mask(jpts, B4A_JF_CZBDSW) & m_neg;
                vad arg = va_sel(m, one - vbd * va_set1(pp->invPhiBSWD), one);
                vad sarg;
                if (!va_any((~(pp->f_mjswd_half ? va_mask_all() : va_mask_zero())) & m))
                    sarg = one / va_sqrt(arg);
                else {
                    sarg = va_exp(-va_set1(pp->MJSWD) * va_log(arg));
                    if (pp->f_mjswd_half && va_any((pp->f_mjswd_half ? va_mask_all() : va_mask_zero()) & m))
                        sarg = va_sel((pp->f_mjswd_half ? va_mask_all() : va_mask_zero()) & m, one / va_sqrt(arg), sarg);
                }
                q_n = va_sel(m, q_n + va_set1(pp->PhiBSWD) * va_jct_gather(jpts, czbdsw) * (one - arg * sarg) * va_set1(pp->inv1mMJSWD), q_n);
                c_n = va_sel(m, c_n + va_jct_gather(jpts, czbdsw) * sarg, c_n);
            }
            if ((jct_any_flags & B4A_JF_CZBDSWG) && va_any(va_jct_mask(jpts, B4A_JF_CZBDSWG) & m_neg)) {
                vam m = va_jct_mask(jpts, B4A_JF_CZBDSWG) & m_neg;
                vad arg = va_sel(m, one - vbd * va_set1(pp->invPhiBSWGD), one);
                vad sarg;
                if (!va_any((~(pp->f_mjswgd_half ? va_mask_all() : va_mask_zero())) & m))
                    sarg = one / va_sqrt(arg);
                else {
                    sarg = va_exp(-va_set1(pp->MJSWGD) * va_log(arg));
                    if (pp->f_mjswgd_half && va_any((pp->f_mjswgd_half ? va_mask_all() : va_mask_zero()) & m))
                        sarg = va_sel((pp->f_mjswgd_half ? va_mask_all() : va_mask_zero()) & m, one / va_sqrt(arg), sarg);
                }
                q_n = va_sel(m, q_n + va_set1(pp->PhiBSWGD) * va_jct_gather(jpts, czbdswg) * (one - arg * sarg) * va_set1(pp->inv1mMJSWGD), q_n);
                c_n = va_sel(m, c_n + va_jct_gather(jpts, czbdswg) * sarg, c_n);
            }
            qbd   = va_sel(m_neg, q_n, qbd);
            capbd = va_sel(m_neg, c_n, capbd);
        }
    }

    /* ===================================================================== */
    /* 11. Overlap Capacitances (capMod == 2) & Total Charge Assembly        */
    /* ===================================================================== */
    vad cgdo, qgdo, cgso, qgso;
    {
        vad T0 = vgd + va_set1(B4V_DELTA_1);
        vad T1 = va_sqrt(T0 * T0 + va_set1(4.0 * B4V_DELTA_1));
        vad T2 = half * (T0 - T1);
        vad T4 = va_sqrt(one - va_set1(4.0) * T2 * va_set1(pp->inv_ckappad));
        cgdo = va_set1(pp->cgdo) + va_set1(pp->weffCV_cgdl) - va_set1(pp->weffCV_cgdl) * (one - one / T4) * (half - half * T0 / T1);
        qgdo = (va_set1(pp->cgdo) + va_set1(pp->weffCV_cgdl)) * vgd - va_set1(pp->weffCV_cgdl) * (T2 + half * va_set1(pp->ckappad) * (T4 - one));

        vad T0s = vgs + va_set1(B4V_DELTA_1);
        vad T1s = va_sqrt(T0s * T0s + va_set1(4.0 * B4V_DELTA_1));
        vad T2s = half * (T0s - T1s);
        vad T4s = va_sqrt(one - va_set1(4.0) * T2s * va_set1(pp->inv_ckappas));
        cgso = va_set1(pp->cgso) + va_set1(pp->weffCV_cgsl) - va_set1(pp->weffCV_cgsl) * (one - one / T4s) * (half - half * T0s / T1s);
        qgso = (va_set1(pp->cgso) + va_set1(pp->weffCV_cgsl)) * vgs - va_set1(pp->weffCV_cgsl) * (T2s + half * va_set1(pp->ckappas) * (T4s - one));

        if (pp->f_nf_ne1) {
            cgdo = cgdo * va_set1(pp->nf);
            cgso = cgso * va_set1(pp->nf);
            qgdo = qgdo * va_set1(pp->nf);
            qgso = qgso * va_set1(pp->nf);
        }
    }

    vad qgb = va_set1(pp->cgbo) * vgb;
    vad qgate_tot = qgate + qgdo + qgso + qgb;
    vad qbulk_tot = qbulk - qgb;
    vad qdrn_fwd  = qdrn - qgdo;
    vad qsrc_fwd  = -(qgate_tot + qbulk_tot + qdrn_fwd);
    vad qsrc_rev  = qdrn - qgso;
    vad qdrn_rev  = -(qgate_tot + qbulk_tot + qsrc_rev);
    vad qdrn_tot  = va_sel(m_fwd, qdrn_fwd, qdrn_rev);
    vad qsrc_tot  = va_sel(m_fwd, qsrc_fwd, qsrc_rev);

    /* Unscaled (time-independent) terminal capacitances */
    vad Cggb = cggb + cgdo + cgso + va_set1(pp->cgbo);
    vad Cgdb = va_sel(m_fwd, cgdb, cgsb) - cgdo;
    vad Cgsb = va_sel(m_fwd, cgsb, cgdb) - cgso;

    vad cgg_bg_dg = cggb + cbgb + cdgb;
    vad Cdgb = va_sel(m_fwd, cdgb - cgdo, -(cgg_bg_dg + cgdo));
    vad Csgb = va_sel(m_fwd, -(cgg_bg_dg + cgso), cdgb - cgso);
    vad Cbgb = cbgb - va_set1(pp->cgbo);

    vad c_sum_d = cgdb + cbdb + cddb;
    vad c_sum_s = cgsb + cbsb + cdsb;
    vad Cddb = va_sel(m_fwd, cddb + capbd + cgdo, capbd + cgdo - c_sum_s);
    vad Cdsb = va_sel(m_fwd, cdsb, -c_sum_d);
    vad Csdb = va_sel(m_fwd, -c_sum_d, cdsb);
    vad Cssb = va_sel(m_fwd, capbs + cgso - c_sum_s, cddb + capbs + cgso);

    vad Cbdb = va_sel(m_fwd, cbdb, cbsb) - capbd;
    vad Cbsb = va_sel(m_fwd, cbsb, cbdb) - capbs;

    /* ===================================================================== */

    vad qg_st = qgate_tot;
    vad qd_st = qdrn_tot - qbd;
    vad qs_st = qsrc_tot - qbs;
    vad qb_st = qbulk_tot + qbd + qbs;

    B4A_SCATTER(l2pts, a_vds, vds);
    B4A_SCATTER(l2pts, a_vgs, vgs);
    B4A_SCATTER(l2pts, a_vbs, vbs);
    B4A_SCATTER(l2pts, a_vbd, vbd);
    B4A_SCATTER(l2pts, qg_st, qg_st);
    B4A_SCATTER(l2pts, qd_st, qd_st);
    B4A_SCATTER(l2pts, qb_st, qb_st);

    if (need_aos) {
        double aqbs[B4A_W], aqbd[B4A_W], aqs[B4A_W];
        va_store(aqbs, qbs); va_store(aqbd, qbd); va_store(aqs, qs_st);
        for (int l = 0; l < nact; l++) {
            int st = cpts[l]->stateBase;
            s0[st + 4]  = l2pts[l]->a_vbs;
            s0[st + 5]  = l2pts[l]->a_vbd;
            s0[st + 6]  = l2pts[l]->a_vbs;
            s0[st + 7]  = l2pts[l]->a_vgs;
            s0[st + 8]  = l2pts[l]->a_vgs;
            s0[st + 9]  = 0.0;
            s0[st + 10] = l2pts[l]->a_vds;
            s0[st + 19] = aqbs[l];
            s0[st + 21] = aqbd[l];
            s0[st + 27] = 0.0;
            s0[st + 28] = aqs[l];
        }
    }

    B4A_SCATTER(l2pts, Cggb, Cggb);
    B4A_SCATTER(l2pts, Cgdb, Cgdb);
    B4A_SCATTER(l2pts, Cgsb, Cgsb);
    B4A_SCATTER(l2pts, Cdgb, Cdgb);
    B4A_SCATTER(l2pts, Cddb, Cddb);
    B4A_SCATTER(l2pts, Cdsb, Cdsb);
    B4A_SCATTER(l2pts, Csgb, Csgb);
    B4A_SCATTER(l2pts, Csdb, Csdb);
    B4A_SCATTER(l2pts, Cssb, Cssb);
    B4A_SCATTER(l2pts, Cbgb, Cbgb);
    B4A_SCATTER(l2pts, Cbdb, Cbdb);
    B4A_SCATTER(l2pts, Cbsb, Cbsb);

    B4A_SCATTER(l2pts, q_lin_g, -Cggb * vgb + Cgdb * vbd + Cgsb * vbs);
    B4A_SCATTER(l2pts, q_lin_d, -Cdgb * vgb + Cddb * vbd + Cdsb * vbs);
    B4A_SCATTER(l2pts, q_lin_b, -Cbgb * vgb + Cbdb * vbd + Cbsb * vbs);

    for (int l = 0; l < nact; l++) {
        bsim4_apple_l2_assemble_one(cpts[l], l2pts[l], spts[l],
                                    l2pts[l]->a_vds, l2pts[l]->a_vgs, l2pts[l]->a_vbs, 0,
                                    ag0, ag1, ag2, is_gear2, is_trap2, s0, s1, s2);
    }
}

void
bsim4_apple_eval_batch(
    const BSIM4ApplePParam * restrict pp,
    const BSIM4AppleJct * const jpts[B4A_W],
    BSIM4AppleCheck * const cpts[B4A_W],
    BSIM4AppleStamp * const spts[B4A_W],
    BSIM4AppleL2 * const l2pts[B4A_W],
    int nact,
    const double vds_in[B4A_W],
    const double vgs_in[B4A_W],
    const double vbs_in[B4A_W],
    double ckt_gmin,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2,
    BSIM4AppleOut * restrict out,
    int need_aos)
{
    BSIM4ApplePipe pipe;

    for (int l = 0; l < nact; l++) {
        cpts[l]->last_vds = vds_in[l];
        cpts[l]->last_vgs = vgs_in[l];
        cpts[l]->last_vbs = vbs_in[l];
    }

    bsim4_apple_p1_vth_mob(pp, &pipe, va_load(vds_in), va_load(vgs_in), va_load(vbs_in));

    double a_von[B4A_W];
    va_store(a_von, pipe.Vth);
    for (int l = 0; l < nact; l++) spts[l]->von = a_von[l];

    bsim4_apple_p2_ids_diode(pp, &pipe, jpts, cpts, spts, l2pts,
                             nact, ckt_gmin, out, need_aos);

    bsim4_apple_p3_cv_jct(pp, &pipe, jpts, cpts, spts, l2pts,
                          nact, ag0, ag1, ag2, is_gear2, is_trap2,
                          s0, s1, s2, need_aos);
}

#undef B4A_SCATTER

#endif /* B4_APPLE_DEFINE_KERNEL */

#endif /* NGSPICE_OPT_APPLE */
#endif /* NGSPICE_BSIM4_APPLE_CORE_H */
