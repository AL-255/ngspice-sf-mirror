/*
 * Apple Silicon / ARM64 NEON SoA evaluator for BSIM4v5 & BSIM4v6 standard
 * 4-terminal transient MOSFETs (rgateMod=0, rbodyMod=0, rdsMod=0, trnqsMod=0,
 * acnqsMod=0, igcMod=0, igbMod=0, mobMod=0, capMod=2, dioMod=1, mtrlMod=0,
 * cvchargeMod=0, tempMod<2, xpart>=0, off=0; supports hasTrapTunnel).
 *
 * Apple Silicon micro-architectural optimizations:
 *   - Default B4N_W=4 (2x float64x2_t saturating Apple M-series 4 FP/NEON pipes
 *     with 2x finer stationary-block cache granularity than 8-wide AVX-512;
 *     also supports B4N_W=2).
 *   - Pure NEON vector predicates (uint64x2_t + bsl/bic/umaxv/uminv) with
 *     zero GPR<->SIMD domain crossings inside the physics kernel.
 *   - Inline branchless IEEE-754 double NEON exp (<= 2 ULP, 4.30e-16) and
 *     128-entry L1-table zero-division NEON log (<= 3 ULP, 4.28e-16, zero fdiv!).
 *   - Live-range split: Diodes + DC matrix/RHS + convTest coefficients computed
 *     and stored BEFORE C-V, freeing 24 vectors (48 NEON registers!) across C-V.
 *   - 3-way 128-byte cache-line-aligned split (BSIM4NeonHot / BSIM4NeonL2 /
 *     BSIM4NeonConst) + lane-interleaved KLU stamps so Level-1 cache hits,
 *     convTest, and trunc NEVER touch L2 or cold model constant cache lines.
 */
#ifndef NGSPICE_BSIM4_NEON_CORE_H
#define NGSPICE_BSIM4_NEON_CORE_H

#if NGSPICE_OPT_APPLE

#include <arm_neon.h>
#include <stdint.h>
#include <math.h>

#ifndef B4N_W
#define B4N_W 4
#endif

#define B4V_MAX_EXP        5.834617425e14
#define B4V_MIN_EXP        1.713908431e-15
#define B4V_EXP_THRESHOLD  34.0
#define B4V_LOG1P_MIN_EXP  1.713908431e-15
#define B4V_EPSSI          1.03594e-10
#define B4V_CHARGE_Q       1.60219e-19
#define B4V_DELTA_1        0.02
#define B4V_DELTA_3        0.02
#define B4V_DELTA_4        0.02

typedef struct { double invc; double hi; double lo; } b4n_log_entry_t;
static const b4n_log_entry_t b4n_log_tab[128] = {
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
static inline float64x2_t b4n_exp2(float64x2_t x) {
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

static inline float64x2_t b4n_log2(float64x2_t x) {
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

    const b4n_log_entry_t *e0 = &b4n_log_tab[i0];
    const b4n_log_entry_t *e1 = &b4n_log_tab[i1];
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

#if B4N_W == 2

typedef float64x2_t vnd;
typedef uint64x2_t  vnm;

static inline vnd vn_set1(double x) { return vdupq_n_f64(x); }
static inline vnd vn_load(const double *p) { return vld1q_f64(p); }
static inline void vn_store(double *p, vnd v) { vst1q_f64(p, v); }
static inline vnd vn_sqrt(vnd x) { return vsqrtq_f64(x); }
static inline vnd vn_abs(vnd x) { return vabsq_f64(x); }
static inline vnd vn_min(vnd a, vnd b) { return vminnmq_f64(a, b); }
static inline vnd vn_max(vnd a, vnd b) { return vmaxnmq_f64(a, b); }
static inline vnm vn_gt(vnd a, vnd b) { return vcgtq_f64(a, b); }
static inline vnm vn_ge(vnd a, vnd b) { return vcgeq_f64(a, b); }
static inline vnm vn_lt(vnd a, vnd b) { return vcltq_f64(a, b); }
static inline vnm vn_le(vnd a, vnd b) { return vcleq_f64(a, b); }
static inline vnm vn_eq(vnd a, vnd b) { return vceqq_f64(a, b); }
static inline vnd vn_sel(vnm m, vnd t, vnd f) { return vbslq_f64(m, t, f); }
static inline vnd vn_exp(vnd x) { return b4n_exp2(x); }
static inline vnd vn_log(vnd x) { return b4n_log2(x); }
static inline int vn_any(vnm m) { return vmaxvq_u32(vreinterpretq_u32_u64(m)) != 0; }
static inline int vn_all(vnm m) { return vminvq_u32(vreinterpretq_u32_u64(m)) != 0; }
static inline vnm vn_mask_zero(void) { return vdupq_n_u64(0); }
static inline vnm vn_mask_from_bits(uint8_t bits) {
    uint64_t arr[2] = { (bits & 1u) ? ~0ULL : 0ULL, (bits & 2u) ? ~0ULL : 0ULL };
    return vld1q_u64(arr);
}
static inline int vn_popcount(vnm m) {
    return (int)(vgetq_lane_u64(m, 0) & 1u) + (int)(vgetq_lane_u64(m, 1) & 1u);
}
static inline int vn_mask_bit(vnm m, int lane) {
    return (lane == 0) ? (vgetq_lane_u64(m, 0) != 0) : (vgetq_lane_u64(m, 1) != 0);
}
static inline void vn_store_lane_slot(double st_lane[B4N_W][20], int slot, vnd v) {
    st_lane[0][slot] = vgetq_lane_f64(v, 0);
    st_lane[1][slot] = vgetq_lane_f64(v, 1);
}

#elif B4N_W == 4

typedef double   vnd __attribute__((vector_size(32), aligned(32)));
typedef uint64_t vnm __attribute__((vector_size(32), aligned(32)));

typedef union { vnd v; float64x2_t q[2]; double d[4]; } b4n_u4d;
typedef union { vnm v; uint64x2_t  q[2]; uint64_t u[4]; } b4n_u4m;

static inline vnd vn_set1(double x) {
    b4n_u4d r; r.q[0] = vdupq_n_f64(x); r.q[1] = vdupq_n_f64(x); return r.v;
}
static inline vnd vn_load(const double *p) {
    b4n_u4d r; r.q[0] = vld1q_f64(p); r.q[1] = vld1q_f64(p + 2); return r.v;
}
static inline void vn_store(double *p, vnd v) {
    b4n_u4d u; u.v = v; vst1q_f64(p, u.q[0]); vst1q_f64(p + 2, u.q[1]);
}
static inline vnd vn_sqrt(vnd x) {
    b4n_u4d u, r; u.v = x; r.q[0] = vsqrtq_f64(u.q[0]); r.q[1] = vsqrtq_f64(u.q[1]); return r.v;
}
static inline vnd vn_abs(vnd x) {
    b4n_u4d u, r; u.v = x; r.q[0] = vabsq_f64(u.q[0]); r.q[1] = vabsq_f64(u.q[1]); return r.v;
}
static inline vnd vn_min(vnd a, vnd b) {
    b4n_u4d ua, ub, r; ua.v = a; ub.v = b;
    r.q[0] = vminnmq_f64(ua.q[0], ub.q[0]); r.q[1] = vminnmq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vnd vn_max(vnd a, vnd b) {
    b4n_u4d ua, ub, r; ua.v = a; ub.v = b;
    r.q[0] = vmaxnmq_f64(ua.q[0], ub.q[0]); r.q[1] = vmaxnmq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vnm vn_gt(vnd a, vnd b) {
    b4n_u4d ua, ub; b4n_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vcgtq_f64(ua.q[0], ub.q[0]); r.q[1] = vcgtq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vnm vn_ge(vnd a, vnd b) {
    b4n_u4d ua, ub; b4n_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vcgeq_f64(ua.q[0], ub.q[0]); r.q[1] = vcgeq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vnm vn_lt(vnd a, vnd b) {
    b4n_u4d ua, ub; b4n_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vcltq_f64(ua.q[0], ub.q[0]); r.q[1] = vcltq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vnm vn_le(vnd a, vnd b) {
    b4n_u4d ua, ub; b4n_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vcleq_f64(ua.q[0], ub.q[0]); r.q[1] = vcleq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vnm vn_eq(vnd a, vnd b) {
    b4n_u4d ua, ub; b4n_u4m r; ua.v = a; ub.v = b;
    r.q[0] = vceqq_f64(ua.q[0], ub.q[0]); r.q[1] = vceqq_f64(ua.q[1], ub.q[1]); return r.v;
}
static inline vnd vn_sel(vnm m, vnd t, vnd f) {
    b4n_u4m um; b4n_u4d ut, uf, r; um.v = m; ut.v = t; uf.v = f;
    r.q[0] = vbslq_f64(um.q[0], ut.q[0], uf.q[0]);
    r.q[1] = vbslq_f64(um.q[1], ut.q[1], uf.q[1]);
    return r.v;
}
static inline vnd vn_exp(vnd x) {
    b4n_u4d u, r; u.v = x; r.q[0] = b4n_exp2(u.q[0]); r.q[1] = b4n_exp2(u.q[1]); return r.v;
}
static inline vnd vn_log(vnd x) {
    b4n_u4d u, r; u.v = x; r.q[0] = b4n_log2(u.q[0]); r.q[1] = b4n_log2(u.q[1]); return r.v;
}
static inline int vn_any(vnm m) {
    b4n_u4m um; um.v = m;
    return vmaxvq_u32(vreinterpretq_u32_u64(vorrq_u64(um.q[0], um.q[1]))) != 0;
}
static inline int vn_all(vnm m) {
    b4n_u4m um; um.v = m;
    return vminvq_u32(vreinterpretq_u32_u64(vandq_u64(um.q[0], um.q[1]))) != 0;
}
static inline vnm vn_mask_zero(void) {
    b4n_u4m r; r.q[0] = vdupq_n_u64(0); r.q[1] = vdupq_n_u64(0); return r.v;
}
static inline vnm vn_mask_from_bits(uint8_t bits) {
    b4n_u4m r;
    for (int i = 0; i < 4; i++) r.u[i] = (bits & (1u << i)) ? ~0ULL : 0ULL;
    return r.v;
}
static inline int vn_popcount(vnm m) {
    b4n_u4m um; um.v = m;
    return (int)(um.u[0] & 1u) + (int)(um.u[1] & 1u) + (int)(um.u[2] & 1u) + (int)(um.u[3] & 1u);
}
static inline int vn_mask_bit(vnm m, int lane) {
    b4n_u4m um; um.v = m; return um.u[lane] != 0;
}
static inline void vn_store_lane_slot(double st_lane[B4N_W][20], int slot, vnd v) {
    b4n_u4d u; u.v = v;
    st_lane[0][slot] = u.d[0];
    st_lane[1][slot] = u.d[1];
    st_lane[2][slot] = u.d[2];
    st_lane[3][slot] = u.d[3];
}

#else
#error "B4N_W must be 2 or 4"
#endif

static inline vnd vn_exp_clamped(vnd x, double lo, double hi) {
    return vn_exp(vn_min(vn_max(x, vn_set1(lo)), vn_set1(hi)));
}

static inline vnd vn_log_pos(vnd x) {
    return vn_log(vn_max(x, vn_set1(1.0e-300)));
}

/* Exact DEVlimvds (matches devsup.c:DEVlimvds) */
static inline vnd vn_limvds(vnd vnew, vnd vold) {
    vnm m_ge35 = vn_ge(vold, vn_set1(3.5));
    vnm m_inc  = vn_gt(vnew, vold);
    vnd hi_inc = vn_min(vnew, vn_set1(3.0) * vold + vn_set1(2.0));
    vnm m_lt35 = vn_lt(vnew, vn_set1(3.5));
    vnd hi_dec = vn_sel(m_lt35, vn_max(vnew, vn_set1(2.0)), vnew);
    vnd hi_res = vn_sel(m_inc, hi_inc, hi_dec);

    vnd lo_inc = vn_min(vnew, vn_set1(4.0));
    vnd lo_dec = vn_max(vnew, vn_set1(-0.5));
    vnd lo_res = vn_sel(m_inc, lo_inc, lo_dec);
    return vn_sel(m_ge35, hi_res, lo_res);
}

/* Exact DEVfetlim (matches devsup.c:DEVfetlim) */
static inline vnd vn_fetlim(vnd vnew, vnd vold, vnd vto) {
    const vnd z0 = vn_set1(0.0);
    vnd diff_o = vold - vto;
    vnd vtsthi = vn_abs(vn_set1(2.0) * diff_o) + vn_set1(2.0);
    vnd vtstlo = vn_abs(diff_o) + vn_set1(1.0);
    vnd vtox   = vto + vn_set1(3.5);
    vnd delv   = vnew - vold;

    vnm m_on  = vn_ge(vold, vto);
    vnm m_hi  = vn_ge(vold, vtox);
    vnm m_dec = vn_le(delv, z0);

    vnm m_n_ge_x = vn_ge(vnew, vtox);
    vnd hi_dec_ge = vn_sel(vn_gt(-delv, vtstlo), vold - vtstlo, vnew);
    vnd hi_dec_lt = vn_max(vnew, vto + vn_set1(2.0));
    vnd hi_dec = vn_sel(m_n_ge_x, hi_dec_ge, hi_dec_lt);
    vnd hi_inc = vn_sel(vn_ge(delv, vtsthi), vold + vtsthi, vnew);
    vnd on_hi  = vn_sel(m_dec, hi_dec, hi_inc);

    vnd on_mid = vn_sel(m_dec, vn_max(vnew, vto - vn_set1(0.5)), vn_min(vnew, vto + vn_set1(4.0)));
    vnd on_res = vn_sel(m_hi, on_hi, on_mid);

    vnd off_dec = vn_sel(vn_gt(-delv, vtsthi), vold - vtsthi, vnew);
    vnd vtemp = vto + vn_set1(0.5);
    vnd off_inc_le = vn_sel(vn_gt(delv, vtstlo), vold + vtstlo, vnew);
    vnd off_inc = vn_sel(vn_le(vnew, vtemp), off_inc_le, vtemp);
    vnd off_res = vn_sel(m_dec, off_dec, off_inc);

    return vn_sel(m_on, on_res, off_res);
}

/* Exact DEVpnjlim (matches devsup.c:DEVpnjlim) */
static inline vnd vn_pnjlim(vnd vnew, vnd vold, double vt_s, vnd vcrit, vnm *limited_mask) {
    const vnd z0 = vn_set1(0.0);
    const vnd vt = vn_set1(vt_s);
    const vnd two = vn_set1(2.0);
    vnm m_fwd = vn_gt(vnew, vcrit) & vn_gt(vn_abs(vnew - vold), vn_set1(vt_s + vt_s));
    vnd res = vnew;
    vnm chk = m_fwd;

    if (vn_any(m_fwd)) {
        vnm m_vo_pos = vn_gt(vold, z0);
        vnd arg = (vnew - vold) * vn_set1(1.0 / vt_s);
        vnm m_arg_pos = vn_gt(arg, z0);
        vnd larg_p = vn_max(arg - two, vn_set1(1.0e-300));
        vnd larg_n = vn_max(two - arg, vn_set1(1.0e-300));
        vnd v_pos = vn_sel(m_arg_pos,
                           vold + vt * (two + vn_log(larg_p)),
                           vold - vt * (two + vn_log(larg_n)));
        vnd v_neg = vt * vn_log(vn_max(vnew * vn_set1(1.0 / vt_s), vn_set1(1.0e-300)));
        vnd v_fwd = vn_sel(m_vo_pos, v_pos, v_neg);
        res = vn_sel(m_fwd, v_fwd, res);
    }

    vnm m_rem_neg = (~m_fwd) & vn_lt(vnew, z0);
    if (vn_any(m_rem_neg)) {
        vnd arg = vn_sel(vn_gt(vold, z0), -vold - vn_set1(1.0), two * vold - vn_set1(1.0));
        vnm m_clamp = m_rem_neg & vn_lt(vnew, arg);
        res = vn_sel(m_clamp, arg, res);
        chk = chk | m_clamp;
    }

    *limited_mask = chk;
    return res;
}

/* Reverse-bias trap-assisted tunneling term (exact match to DEXP) */
static inline void
vn_trap_tunnel_term(vnm mask, vnd vj, vnd vt, vnd invNvtm, vnd revSat,
                    vnd *g_io, vnd *c_io)
{
    const vnd one = vn_set1(1.0);
    const vnd exp_th = vn_set1(B4V_EXP_THRESHOLD);
    const vnd nexp_th = vn_set1(-B4V_EXP_THRESHOLD);
    const vnd max_exp = vn_set1(B4V_MAX_EXP);
    const vnd min_exp = vn_set1(B4V_MIN_EXP);

    vnd diff = vt - vj;
    vnm m_small = vn_lt(diff, vt * vn_set1(1.0e-3));
    vnd T9_reg = one / vn_sel(m_small, one, diff);
    vnd T0_sm  = -vj * invNvtm * vn_set1(1.0e3);
    vnd T0_reg = -vj * invNvtm * vt * T9_reg;
    vnd T0     = vn_sel(m_small, T0_sm, T0_reg);

    vnm m_hi  = vn_gt(T0, exp_th);
    vnm m_lo  = vn_lt(T0, nexp_th);
    vnm m_mid = (~m_hi) & (~m_lo) & mask;

    vnd B = vn_sel(m_hi, max_exp * (one + T0 - exp_th), min_exp);
    vnd C = vn_sel(m_hi, max_exp, vn_set1(0.0));
    if (vn_any(m_mid)) {
        vnd ex = vn_exp_clamped(T0, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
        B = vn_sel(m_mid, ex, B);
        C = vn_sel(m_mid, ex, C);
    }

    vnd dT0_sm  = invNvtm * vn_set1(1.0e3);
    vnd dT0_reg = vt * invNvtm * (T9_reg + vj * T9_reg * T9_reg);
    vnd dB_dVb  = C * vn_sel(m_small, dT0_sm, dT0_reg);

    *g_io = vn_sel(mask, *g_io + revSat * dB_dVb, *g_io);
    *c_io = vn_sel(mask, *c_io - revSat * (B - one), *c_io);
}

/* Poly gate depletion (exact match to BSIM4v5/v6 polyDepletion with mtrlMod=0) */
static inline void
vn_poly_depletion(vnd phi_vfb, vnd polyT1, vnm m_ngate_valid,
                  vnd Vgs, vnd *Vgs_eff, vnd *dVgs_eff_dVg)
{
    vnm m = m_ngate_valid & vn_gt(Vgs, phi_vfb);
    if (!vn_any(m)) {
        *Vgs_eff = Vgs;
        *dVgs_eff_dVg = vn_set1(1.0);
        return;
    }
    vnd one = vn_set1(1.0);
    vnd T8 = vn_max(Vgs - phi_vfb, vn_set1(0.0));
    vnd invT1 = one / polyT1;
    vnd T4 = vn_sqrt(one + vn_set1(2.0) * T8 * invT1);
    vnd T2 = vn_set1(2.0) * T8 / (T4 + one);
    vnd T3 = vn_set1(0.5) * T2 * T2 * invT1;
    vnd T7 = vn_set1(1.07) - T3;
    vnd T6 = vn_sqrt(T7 * T7 + vn_set1(0.224));
    vnd T5 = vn_set1(1.12) - vn_set1(0.5) * (T7 + T6);
    vnd eff = Vgs - T5;
    vnd deff = one - (vn_set1(0.5) - vn_set1(0.5) / T4) * (one + T7 / T6);
    *Vgs_eff = vn_sel(m, eff, Vgs);
    *dVgs_eff_dVg = vn_sel(m, deff, one);
}

/* ========================================================================= */
/* 3-Way 128-Byte Cache-Line-Aligned SoA Structures for Apple Silicon        */
/* ========================================================================= */

/*
 * 1. HOT block: Level-1 cache check, direct KLU/Sparse stamping, convTest, trunc.
 *    Level-1 stationary hits NEVER touch BSIM4NeonL2 or BSIM4NeonConst!
 */
typedef struct __attribute__((aligned(128))) {
    /* First 128B cache line(s): Level-1 check + convTest + limiting */
    vnd type;
    vnd von;
    vnd vcrit_v;
    vnd last_vds, last_vgs, last_vbs;
    vnd Idtot, cd_g, cd_d, cd_b;
    vnd Ibtot, cb_g, cb_d, cb_b;
    vnm active_mask;

    /* Cache validity metadata */
    double last_time;
    double last_ag0, last_ag1, last_ag2;
    double last_gmin;
    const double *last_s0;
    const double *last_s1;
    const double *last_s2;
    int last_int_mode;
    uint8_t n_active;
    uint8_t m_has_dint;
    uint8_t m_has_sint;
    uint8_t phys_valid;
    uint8_t stamp_valid;

    /* Per-lane scalar gather/scatter metadata */
    int dNodePrime[B4N_W];
    int gNodePrime[B4N_W];
    int sNodePrime[B4N_W];
    int bNodePrime[B4N_W];
    int stateBase[B4N_W];
    int instIdx[B4N_W];
    double mgd[B4N_W];
    double mgs[B4N_W];

    /* Lane-interleaved Level-1 cached matrix/RHS stamps (160B contiguous per lane!)
     * Order: [0..3]=rhsd,rhsg,rhsb,rhss, [4..7]=v34..37, [8..11]=v44,46,47,48,
     *        [12..15]=v51,52,53,55, [16..19]=v58..61 */
    double st_lane[B4N_W][20];
    int kluOff[B4N_W][22];
} BSIM4NeonHot;

/*
 * 2. LEVEL-2 block: time-independent DC + capacitance + charge physics cache.
 *    Touched on Level-2 hits and full evals (and rare nonzero-dv Level-1 sync).
 */
typedef struct __attribute__((aligned(128))) {
    vnd m;
    vnd mtype;
    vnd Cggb, Cgdb, Cgsb;
    vnd Cdgb, Cddb, Cdsb;
    vnd Csgb, Csdb, Cssb;
    vnd Cbgb, Cbdb, Cbsb;
    vnd mdc44, mdc46, mdc47, mdc48;
    vnd mdc51, mdc52, mdc53, mdc55;
    vnd mdc58, mdc59, mdc60, mdc61;
    vnd mdc_rhsd, mdc_rhsb, mdc_rhss;
    vnd q_lin_g, q_lin_d, q_lin_b;
    vnd qb_st, qg_st, qd_st;
    double a_vbd[B4N_W], a_vbs[B4N_W], a_vgs[B4N_W], a_vds[B4N_W];
    double a_qb[B4N_W], a_qg[B4N_W], a_qd[B4N_W];
} BSIM4NeonL2;

/*
 * 3. COLD block: model/geometry BSIM4 physics constants (touched ONLY on full physics eval!).
 */
typedef struct __attribute__((aligned(128))) {
    vnd type, m, nf;
    vnd vth0_signed, vfb_plus_phi, vfbzb, vbsc;
    vnd phi, phi_095, sqrtPhi, V0, vtm;
    vnd coxe, invCoxe, coxp, toxe, invToxe, toxp;
    vnd tox_ctm, inv_tox_ctm, inv_2e8_toxp, link_ctm;
    vnd leff, invLeff, sqrtLeff, weff, weffCV, leffCV;
    vnd CoxWL, CoxWL_ov_coxe, drainCond, sourceCond;
    vnd ados1p9, bdos0p7;

    /* Diode (dioMod == 1) & Trap-Assisted Tunneling */
    vnd invNvtms, invNvtmd, SourceSatCurrent, DrainSatCurrent;
    vnd vjsmFwd, vjdmFwd, IVjsmFwd, IVjdmFwd;
    vnd invNvtmrs_s, invNvtmrs_d, invNvtmrssw_s, invNvtmrssw_d, invNvtmrsswg_s, invNvtmrsswg_d;
    vnd vtss, vtsd, vtssws, vtsswd, vtsswgs, vtsswgd;
    vnd SjctRevSat, DjctRevSat, SswRevSat, DswRevSat, SswgRevSat, DswgRevSat;

    /* Vth / SCE / DIBL / n / Pocket */
    vnd xdep0OverSqrtPhi, factor1;
    vnd dvt0, dvt1_leff, dvt2, dvt0w, dvt1w_weff_leff, dvt2w;
    vnd k1, k1ox, k2ox, k3, k3b, lpebFactor, vthNarrowW, vth_const_T1, kt2_TempRatio;
    vnd eta0, etab, theta0vb0, nfactor, cit, cdsc, cdscb, cdscd, dvtp0, dvtp1;

    /* Poly depletion / Vgsteff */
    vnd polyT1, mstar, voffcbn, coxe_ov_cdep0;

    /* Weff / Rds / Abulk */
    vnd dwg, dwb, prwg, prwb, rds0_half, rdswmin, xj, a0;
    vnd b0_ov_wb1, ags_a0, keta, abulkCVfactor;

    /* Mobility / Saturation / CLM / DIBL / DITS / SCBE / Isub / Fsevl */
    vnd ua, ub, uc, ud, u0temp, vsattemp, a1, a2, delta;
    vnd lambda, litl, fprout, pvag, pclm, inv_pclm_litl;
    vnd thetaRout, inv_thetaRout, pdiblb, pdits, inv_pdits, pditsd, pditsl_term;
    vnd pscbe1_litl, inv_pscbe2, alpha_ov_leff, beta0, inv_vtl_tfactor, vtfbphi2;

    /* GIDL / GISL */
    vnd inv_3toxe, agidl_weffCJ, bgidl, cgidl, egidl;
    vnd agisl_weffCJ, bgisl, cgisl, egisl;

    /* C-V (capMod == 2) & Overlap */
    vnd noff, voffcv, acde, ldeb, moin, cgdo, cgso, cgbo;
    vnd weffCV_cgdl, ckappad, inv_ckappad, weffCV_cgsl, ckappas, inv_ckappas;

    /* Junction C-V */
    vnd czbs, czbssw, czbsswg, czbd, czbdsw, czbdswg;
    vnd PhiBS, invPhiBS, PhiBSWS, invPhiBSWS, PhiBSWGS, invPhiBSWGS;
    vnd PhiBD, invPhiBD, PhiBSWD, invPhiBSWD, PhiBSWGD, invPhiBSWGD;
    vnd MJS, inv1mMJS, MJSWS, inv1mMJSWS, MJSWGS, inv1mMJSWGS;
    vnd MJD, inv1mMJD, MJSWD, inv1mMJSWD, MJSWGD, inv1mMJSWGD;
    vnd jctS_sum, jctD_sum, jctS_lin, jctD_lin;

    /* Vector masks (pre-expanded into NEON predicates -> zero GPR->SIMD transfers) */
    vnm m_ngate_valid, m_dvt0w, m_dvtp0, m_fprout, m_pclm;
    vnm m_thetaRout, m_pdits, m_pscbe, m_isub, m_vtl;
    vnm m_lambda, m_gidl_en, m_gisl_en, m_xpart_gt05, m_xpart_lt05;
    vnm m_src_sat_pos, m_drn_sat_pos;
    vnm m_czbs, m_czbssw, m_czbsswg, m_czbd, m_czbdsw, m_czbdswg;
    vnm m_mjs_half, m_mjsws_half, m_mjswgs_half;
    vnm m_mjd_half, m_mjswd_half, m_mjswgd_half;
    vnm m_k1ox_zero, m_k1ox_le0, m_vfbzb_le0, m_nf_ne1, m_has_a1;
    vnm m_tt_s, m_tt_d, m_tt_ssw, m_tt_dsw, m_tt_sswg, m_tt_dswg;

    /* Scalar 1-byte fast-skip flags (zero NEON reduction cost for `if (b->f_...)`) */
    uint8_t f_ngate_valid, f_dvt0w, f_dvtp0, f_fprout, f_pclm;
    uint8_t f_thetaRout, f_pdits, f_pscbe, f_isub, f_vtl;
    uint8_t f_lambda, f_gidl_en, f_gisl_en, f_xpart_gt05, f_xpart_lt05;
    uint8_t f_src_sat_pos, f_drn_sat_pos;
    uint8_t f_czbs, f_czbssw, f_czbsswg, f_czbd, f_czbdsw, f_czbdswg;
    uint8_t f_mjs_half, f_mjsws_half, f_mjswgs_half;
    uint8_t f_mjd_half, f_mjswd_half, f_mjswgd_half;
    uint8_t f_k1ox_zero, f_k1ox_le0, f_vfbzb_le0, f_nf_ne1, f_has_a1;
    uint8_t f_tt_s, f_tt_d, f_tt_ssw, f_tt_dsw, f_tt_sswg, f_tt_dswg;

    /* Block-level summary flags */
    uint8_t is_v6;
    uint8_t has_ud;
    uint8_t xpart_mode;
    uint8_t has_trap_tunnel;
} BSIM4NeonConst;

typedef struct __attribute__((aligned(64))) {
    double von[B4N_W];
    int    mode[B4N_W];
    double cd[B4N_W], gds[B4N_W], gm[B4N_W], gmbs[B4N_W];
    double gbd[B4N_W], gbs[B4N_W], cbs[B4N_W], cbd[B4N_W];
    double csub[B4N_W], gbds[B4N_W], gbgs[B4N_W], gbbs[B4N_W];
    double Igidl[B4N_W], ggidld[B4N_W], ggidlg[B4N_W], ggidlb[B4N_W];
    double Igisl[B4N_W], ggisls[B4N_W], ggislg[B4N_W], ggislb[B4N_W];
} BSIM4NeonOut;

/*
 * Fast Level-1 state sync helper (first-order Taylor dq = C * dv).
 */
static inline void
bsim4_neon_l1_sync_state(
    const BSIM4NeonHot * restrict hot,
    const BSIM4NeonL2 * restrict l2,
    vnd vds, vnd vgs, vnd vbs,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2)
{
    vnd dvds = vds - hot->last_vds;
    vnd dvgs = vgs - hot->last_vgs;
    vnd dvbs = vbs - hot->last_vbs;
    vnd dvgb = dvgs - dvbs;
    vnd dvbd = dvbs - dvds;

    vnd qg_new = l2->qg_st + (l2->Cggb * dvgb - l2->Cgdb * dvbd - l2->Cgsb * dvbs);
    vnd qd_new = l2->qd_st + (l2->Cdgb * dvgb - l2->Cddb * dvbd - l2->Cdsb * dvbs);
    vnd qb_new = l2->qb_st + (l2->Cbgb * dvgb - l2->Cbdb * dvbd - l2->Cbsb * dvbs);

    double lvds[B4N_W], lvgs[B4N_W], lvbs[B4N_W], lvbd[B4N_W];
    double lqb[B4N_W], lqg[B4N_W], lqd[B4N_W];
    vn_store(lvds, vds); vn_store(lvgs, vgs);
    vn_store(lvbs, vbs); vn_store(lvbd, vbs - vds);
    vn_store(lqb, qb_new); vn_store(lqg, qg_new); vn_store(lqd, qd_new);

    const int nact = hot->n_active;
    for (int lane = 0; lane < nact; lane++) {
        int st = hot->stateBase[lane];
        double qb = lqb[lane], qg = lqg[lane], qd = lqd[lane];
        s0[st + 0]  = lvbd[lane];
        s0[st + 1]  = lvbs[lane];
        s0[st + 2]  = lvgs[lane];
        s0[st + 3]  = lvds[lane];
        s0[st + 11] = qb;
        s0[st + 13] = qg;
        s0[st + 15] = qd;

        double cqb, cqg, cqd;
        if (is_gear2) {
            cqb = ag2 * s2[st + 11] + ag1 * s1[st + 11] + ag0 * qb;
            cqg = ag2 * s2[st + 13] + ag1 * s1[st + 13] + ag0 * qg;
            cqd = ag2 * s2[st + 15] + ag1 * s1[st + 15] + ag0 * qd;
        } else if (is_trap2) {
            cqb = -s1[st + 12] * ag1 + ag0 * (qb - s1[st + 11]);
            cqg = -s1[st + 14] * ag1 + ag0 * (qg - s1[st + 13]);
            cqd = -s1[st + 16] * ag1 + ag0 * (qd - s1[st + 15]);
        } else {
            cqb = ag0 * qb + ag1 * s1[st + 11];
            cqg = ag0 * qg + ag1 * s1[st + 13];
            cqd = ag0 * qd + ag1 * s1[st + 15];
        }
        s0[st + 12] = cqb;
        s0[st + 14] = cqg;
        s0[st + 16] = cqd;
    }
}

/*
 * Fast Level-2 helper: integrate cached charges and assemble lane-interleaved
 * matrix/RHS stamps in hot->st_lane[lane][0..19]. NEVER touches BSIM4NeonConst!
 */
static inline void
bsim4_neon_integrate_and_assemble(
    BSIM4NeonHot * restrict hot,
    const BSIM4NeonL2 * restrict l2,
    vnd vds, vnd vgs, vnd vbs,
    int has_dv,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2)
{
    double a_cqb[B4N_W], a_cqg[B4N_W], a_cqd[B4N_W];
    const int nact = hot->n_active;
    vnd dqg = vn_set1(0.0), dqd = vn_set1(0.0), dqb = vn_set1(0.0);

    if (has_dv) {
        vnd dvds = vds - hot->last_vds;
        vnd dvgs = vgs - hot->last_vgs;
        vnd dvbs = vbs - hot->last_vbs;
        vnd dvgb = dvgs - dvbs;
        vnd dvbd = dvbs - dvds;
        dqg = l2->Cggb * dvgb - l2->Cgdb * dvbd - l2->Cgsb * dvbs;
        dqd = l2->Cdgb * dvgb - l2->Cddb * dvbd - l2->Cdsb * dvbs;
        dqb = l2->Cbgb * dvgb - l2->Cbdb * dvbd - l2->Cbsb * dvbs;

        double lvds[B4N_W], lvgs[B4N_W], lvbs[B4N_W], lvbd[B4N_W];
        double lqb[B4N_W], lqg[B4N_W], lqd[B4N_W];
        vn_store(lvds, vds); vn_store(lvgs, vgs);
        vn_store(lvbs, vbs); vn_store(lvbd, vbs - vds);
        vn_store(lqb, l2->qb_st + dqb);
        vn_store(lqg, l2->qg_st + dqg);
        vn_store(lqd, l2->qd_st + dqd);

        for (int lane = 0; lane < nact; lane++) {
            int st = hot->stateBase[lane];
            double qb = lqb[lane], qg = lqg[lane], qd = lqd[lane];
            s0[st + 0]  = lvbd[lane];
            s0[st + 1]  = lvbs[lane];
            s0[st + 2]  = lvgs[lane];
            s0[st + 3]  = lvds[lane];
            s0[st + 11] = qb;
            s0[st + 13] = qg;
            s0[st + 15] = qd;

            double cqb, cqg, cqd;
            if (is_gear2) {
                cqb = ag2 * s2[st + 11] + ag1 * s1[st + 11] + ag0 * qb;
                cqg = ag2 * s2[st + 13] + ag1 * s1[st + 13] + ag0 * qg;
                cqd = ag2 * s2[st + 15] + ag1 * s1[st + 15] + ag0 * qd;
            } else if (is_trap2) {
                cqb = -s1[st + 12] * ag1 + ag0 * (qb - s1[st + 11]);
                cqg = -s1[st + 14] * ag1 + ag0 * (qg - s1[st + 13]);
                cqd = -s1[st + 16] * ag1 + ag0 * (qd - s1[st + 15]);
            } else {
                cqb = ag0 * qb + ag1 * s1[st + 11];
                cqg = ag0 * qg + ag1 * s1[st + 13];
                cqd = ag0 * qd + ag1 * s1[st + 15];
            }
            s0[st + 12] = a_cqb[lane] = cqb;
            s0[st + 14] = a_cqg[lane] = cqg;
            s0[st + 16] = a_cqd[lane] = cqd;
        }
    } else {
        for (int lane = 0; lane < nact; lane++) {
            int st = hot->stateBase[lane];
            double qb = l2->a_qb[lane];
            double qg = l2->a_qg[lane];
            double qd = l2->a_qd[lane];
            s0[st + 0]  = l2->a_vbd[lane];
            s0[st + 1]  = l2->a_vbs[lane];
            s0[st + 2]  = l2->a_vgs[lane];
            s0[st + 3]  = l2->a_vds[lane];
            s0[st + 11] = qb;
            s0[st + 13] = qg;
            s0[st + 15] = qd;

            double cqb, cqg, cqd;
            if (is_gear2) {
                cqb = ag2 * s2[st + 11] + ag1 * s1[st + 11] + ag0 * qb;
                cqg = ag2 * s2[st + 13] + ag1 * s1[st + 13] + ag0 * qg;
                cqd = ag2 * s2[st + 15] + ag1 * s1[st + 15] + ag0 * qd;
            } else if (is_trap2) {
                cqb = -s1[st + 12] * ag1 + ag0 * (qb - s1[st + 11]);
                cqg = -s1[st + 14] * ag1 + ag0 * (qg - s1[st + 13]);
                cqd = -s1[st + 16] * ag1 + ag0 * (qd - s1[st + 15]);
            } else {
                cqb = ag0 * qb + ag1 * s1[st + 11];
                cqg = ag0 * qg + ag1 * s1[st + 13];
                cqd = ag0 * qd + ag1 * s1[st + 15];
            }
            s0[st + 12] = a_cqb[lane] = cqb;
            s0[st + 14] = a_cqg[lane] = cqg;
            s0[st + 16] = a_cqd[lane] = cqd;
        }
    }
    for (int lane = nact; lane < B4N_W; lane++) {
        a_cqb[lane] = a_cqg[lane] = a_cqd[lane] = 0.0;
    }

    vnd v_ag0 = vn_set1(ag0);
    vnd vm_ag0 = l2->m * v_ag0;
    vnd cqbody = vn_load(a_cqb);
    vnd cqgate = vn_load(a_cqg);
    vnd cqdrn  = vn_load(a_cqd);

    vnd mtype = l2->mtype;
    vnd mceqqg = (cqgate + v_ag0 * (l2->q_lin_g - dqg)) * mtype;
    vnd mceqqd = (cqdrn  + v_ag0 * (l2->q_lin_d - dqd)) * mtype;
    vnd mceqqb = (cqbody + v_ag0 * (l2->q_lin_b - dqb)) * mtype;

    vn_store_lane_slot(hot->st_lane, 0, l2->mdc_rhsd - mceqqd);
    vn_store_lane_slot(hot->st_lane, 1, mceqqg);
    vn_store_lane_slot(hot->st_lane, 2, l2->mdc_rhsb - mceqqb);
    vn_store_lane_slot(hot->st_lane, 3, l2->mdc_rhss + mceqqg + mceqqb + mceqqd);

    vnd v34 = l2->Cggb * vm_ag0;
    vnd v35 = l2->Cgdb * vm_ag0;
    vnd v36 = l2->Cgsb * vm_ag0;
    vn_store_lane_slot(hot->st_lane, 4, v34);
    vn_store_lane_slot(hot->st_lane, 5, v35);
    vn_store_lane_slot(hot->st_lane, 6, v36);
    vn_store_lane_slot(hot->st_lane, 7, -(v34 + v35 + v36));

    vnd mgcdgb = l2->Cdgb * vm_ag0;
    vnd mgcddb = l2->Cddb * vm_ag0;
    vnd mgcdsb = l2->Cdsb * vm_ag0;
    vn_store_lane_slot(hot->st_lane, 8,  l2->mdc44 + mgcddb);
    vn_store_lane_slot(hot->st_lane, 9,  l2->mdc46 + mgcdgb);
    vn_store_lane_slot(hot->st_lane, 10, l2->mdc47 - mgcdsb);
    vn_store_lane_slot(hot->st_lane, 11, l2->mdc48 + (mgcdgb + mgcddb + mgcdsb));

    vnd mgcsgb = l2->Csgb * vm_ag0;
    vnd mgcsdb = l2->Csdb * vm_ag0;
    vnd mgcssb = l2->Cssb * vm_ag0;
    vn_store_lane_slot(hot->st_lane, 12, l2->mdc51 - mgcsdb);
    vn_store_lane_slot(hot->st_lane, 13, l2->mdc52 + mgcsgb);
    vn_store_lane_slot(hot->st_lane, 14, l2->mdc53 + mgcssb);
    vn_store_lane_slot(hot->st_lane, 15, l2->mdc55 + (mgcsgb + mgcsdb + mgcssb));

    vnd mgcbgb = l2->Cbgb * vm_ag0;
    vnd mgcbdb = l2->Cbdb * vm_ag0;
    vnd mgcbsb = l2->Cbsb * vm_ag0;
    vn_store_lane_slot(hot->st_lane, 16, l2->mdc58 + mgcbdb);
    vn_store_lane_slot(hot->st_lane, 17, l2->mdc59 + mgcbgb);
    vn_store_lane_slot(hot->st_lane, 18, l2->mdc60 + mgcbsb);
    vn_store_lane_slot(hot->st_lane, 19, l2->mdc61 - (mgcbgb + mgcbdb + mgcbsb));
}

/*
 * Full NEON physics evaluation of B4N_W MOSFETs in parallel.
 */
static inline void
bsim4_neon_eval_block(
    const BSIM4NeonConst * restrict b,
    BSIM4NeonHot * restrict hot,
    BSIM4NeonL2 * restrict l2,
    vnd vds, vnd vgs, vnd vbs,
    double ckt_gmin,
    double ag0, double ag1, double ag2,
    int is_gear2, int is_trap2,
    double * restrict s0,
    const double * restrict s1,
    const double * restrict s2,
    BSIM4NeonOut * restrict out,
    int need_aos)
{
    const vnd z0 = vn_set1(0.0);
    const vnd one = vn_set1(1.0);
    const vnd half = vn_set1(0.5);
    const vnd two = vn_set1(2.0);
    const vnd gmin = vn_set1(ckt_gmin);
    const vnd max_exp = vn_set1(B4V_MAX_EXP);
    const vnd min_exp = vn_set1(B4V_MIN_EXP);
    const vnd exp_th = vn_set1(B4V_EXP_THRESHOLD);
    const vnd nexp_th = vn_set1(-B4V_EXP_THRESHOLD);
    const vnd epssi = vn_set1(B4V_EPSSI);

    vnd vbd = vbs - vds;
    vnd vgd = vgs - vds;
    vnd vgb = vgs - vbs;

    hot->last_vds = vds;
    hot->last_vgs = vgs;
    hot->last_vbs = vbs;

    /* ===================================================================== */

    /* === PHASE A: DC Physics + Diodes + DC Stamp & ConvTest Assembly === */

    /* 2. Forward / Reverse mode swap                                        */
    /* ===================================================================== */
    vnm m_fwd = vn_ge(vds, z0);
    vnd Vds = vn_sel(m_fwd, vds, -vds);
    vnd Vgs = vn_sel(m_fwd, vgs, vgd);
    vnd Vbs = vn_sel(m_fwd, vbs, vbd);

    /* ===================================================================== */
    /* 3. Vbseff, Phis, Xdep, Vth                                            */
    /* ===================================================================== */
    vnd Vbseff, dVbseff_dVb;
    {
        vnd T0 = Vbs - b->vbsc - vn_set1(0.001);
        vnd T1 = vn_sqrt(T0 * T0 - vn_set1(0.004) * b->vbsc);
        vnm m_ge0 = vn_ge(T0, z0);
        vnd v_pos = b->vbsc + half * (T0 + T1);
        vnd dv_pos = half * (one + T0 / T1);
        vnd T2 = vn_set1(-0.002) / (T1 - T0);
        vnd v_neg = b->vbsc * (one + T2);
        vnd dv_neg = T2 * b->vbsc / T1;
        Vbseff = vn_sel(m_ge0, v_pos, v_neg);
        dVbseff_dVb = vn_sel(m_ge0, dv_pos, dv_neg);

        vnd T9 = b->phi_095;
        T0 = T9 - Vbseff - vn_set1(0.001);
        T1 = vn_sqrt(T0 * T0 + vn_set1(0.004) * T9);
        Vbseff = T9 - half * (T0 + T1);
        dVbseff_dVb = dVbseff_dVb * (half * (one + T0 / T1));
    }

    vnd Phis = b->phi - Vbseff;
    vnd sqrtPhis = vn_sqrt(Phis);
    vnd dsqrtPhis_dVb = vn_set1(-0.5) / sqrtPhis;
    vnd Xdep = b->xdep0OverSqrtPhi * sqrtPhis;
    vnd dXdep_dVb = b->xdep0OverSqrtPhi * dsqrtPhis_dVb;

    vnd sqrtXdep = vn_sqrt(Xdep);
    vnd invSqrtXdep = one / sqrtXdep;

    /* lt1 */
    vnd lt1, dlt1_dVb;
    {
        vnd T0 = b->dvt2 * Vbseff;
        vnm m_ge = vn_ge(T0, vn_set1(-0.5));
        vnd T4 = one / (vn_set1(3.0) + vn_set1(8.0) * T0);
        vnd T1 = vn_sel(m_ge, one + T0, (one + vn_set1(3.0) * T0) * T4);
        vnd T2 = vn_sel(m_ge, b->dvt2, b->dvt2 * T4 * T4);
        lt1 = b->factor1 * sqrtXdep * T1;
        dlt1_dVb = b->factor1 * (half * invSqrtXdep * T1 * dXdep_dVb + sqrtXdep * T2);
    }

    /* Theta0 */
    vnd Theta0, dTheta0_dVb;
    {
        vnd T0 = b->dvt1_leff / lt1;
        vnm m_lt = vn_lt(T0, exp_th);
        vnd th_hi = vn_set1(1.0 / (B4V_MAX_EXP - 2.0));
        if (vn_any(m_lt)) {
            vnd T1 = vn_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            vnd T2 = T1 - one;
            vnd T4 = T2 * T2 + two * T1 * min_exp;
            vnd invT4 = one / T4;
            vnd th = T1 * invT4;
            vnd dT1 = -T0 * T1 * dlt1_dVb / lt1;
            vnd dth = dT1 * (T4 - two * T1 * (T2 + min_exp)) * (invT4 * invT4);
            Theta0 = vn_sel(m_lt, th, th_hi);
            dTheta0_dVb = vn_sel(m_lt, dth, z0);
        } else {
            Theta0 = th_hi;
            dTheta0_dVb = z0;
        }
    }
    vnd Delt_vth = b->dvt0 * Theta0 * b->V0;
    vnd dDelt_vth_dVb = b->dvt0 * dTheta0_dVb * b->V0;

    /* dvt0w narrow-width SCE (skip completely if all lanes have dvt0w == 0) */
    vnd T2_dvt0w = z0, dT2_dvt0w_dVb = z0;
    if (b->f_dvt0w) {
        vnd T0w = b->dvt2w * Vbseff;
        vnm m_ge = vn_ge(T0w, vn_set1(-0.5));
        vnd T4w = one / (vn_set1(3.0) + vn_set1(8.0) * T0w);
        vnd T1w = vn_sel(m_ge, one + T0w, (one + vn_set1(3.0) * T0w) * T4w);
        vnd T2w = vn_sel(m_ge, b->dvt2w, b->dvt2w * T4w * T4w);
        vnd ltw = b->factor1 * sqrtXdep * T1w;
        vnd dltw_dVb = b->factor1 * (half * invSqrtXdep * T1w * dXdep_dVb + sqrtXdep * T2w);
        vnd T0 = b->dvt1w_weff_leff / ltw;
        vnm m_lt = vn_lt(T0, exp_th) & b->m_dvt0w;
        vnd t5_hi = vn_set1(1.0 / (B4V_MAX_EXP - 2.0));
        vnd T5 = t5_hi, dT5_dVb = z0;
        if (vn_any(m_lt)) {
            vnd T1 = vn_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            vnd T2 = T1 - one;
            vnd T4 = T2 * T2 + two * T1 * min_exp;
            vnd invT4 = one / T4;
            vnd dT1 = -T0 * T1 * dltw_dVb / ltw;
            T5 = vn_sel(m_lt, T1 * invT4, t5_hi);
            dT5_dVb = vn_sel(m_lt, dT1 * (T4 - two * T1 * (T2 + min_exp)) * (invT4 * invT4), z0);
        }
        T2_dvt0w = vn_sel(b->m_dvt0w, b->dvt0w * T5 * b->V0, z0);
        dT2_dvt0w_dVb = vn_sel(b->m_dvt0w, b->dvt0w * dT5_dVb * b->V0, z0);
    }

    /* DIBL shift */
    vnd dDIBL_Sft_dVd, DIBL_Sft, T4_dibl;
    {
        vnd T3 = b->eta0 + b->etab * Vbseff;
        vnm m_lt = vn_lt(T3, vn_set1(1.0e-4));
        vnd T9 = one / (vn_set1(3.0) - vn_set1(2.0e4) * T3);
        T3 = vn_sel(m_lt, (vn_set1(2.0e-4) - T3) * T9, T3);
        T4_dibl = vn_sel(m_lt, T9 * T9, one);
        dDIBL_Sft_dVd = T3 * b->theta0vb0;
        DIBL_Sft = dDIBL_Sft_dVd * Vds;
    }

    vnd Vth = b->vth_const_T1 + b->k1ox * sqrtPhis * b->lpebFactor
            - b->k2ox * Vbseff - Delt_vth - T2_dvt0w
            + b->k3b * Vbseff * b->vthNarrowW
            + b->kt2_TempRatio * Vbseff - DIBL_Sft;
    vnd dVth_dVb = b->lpebFactor * b->k1ox * dsqrtPhis_dVb - b->k2ox
                 - dDelt_vth_dVb - dT2_dvt0w_dVb + b->k3b * b->vthNarrowW
                 - b->etab * Vds * b->theta0vb0 * T4_dibl + b->kt2_TempRatio;
    vnd dVth_dVd = -dDIBL_Sft_dVd;

    /* Subthreshold swing n */
    vnd n, dn_dVb, dn_dVd;
    {
        vnd tmp1 = epssi / Xdep;
        vnd tmp2 = b->nfactor * tmp1;
        vnd tmp3 = b->cdsc + b->cdscb * Vbseff + b->cdscd * Vds;
        vnd tmp4 = (tmp2 + tmp3 * Theta0 + b->cit) * b->invCoxe;
        vnd base_dVb = (-tmp2 / Xdep * dXdep_dVb + tmp3 * dTheta0_dVb + b->cdscb * Theta0) * b->invCoxe;
        vnd base_dVd = b->cdscd * Theta0 * b->invCoxe;
        vnm m_ge = vn_ge(tmp4, vn_set1(-0.5));
        vnd T0 = one / (vn_set1(3.0) + vn_set1(8.0) * tmp4);
        vnd T0sq = T0 * T0;
        n = vn_sel(m_ge, one + tmp4, (one + vn_set1(3.0) * tmp4) * T0);
        dn_dVb = vn_sel(m_ge, base_dVb, base_dVb * T0sq);
        dn_dVd = vn_sel(m_ge, base_dVd, base_dVd * T0sq);
    }

    /* Pocket implant Vth correction (dvtp0 > 0) */
    if (b->f_dvtp0) {
        vnd T0 = -b->dvtp1 * Vds;
        vnm m_ge = vn_ge(T0, nexp_th) & b->m_dvtp0;
        vnd T2 = min_exp, dT2_dVd = z0;
        if (vn_any(m_ge)) {
            vnd ex = vn_exp_clamped(T0, -B4V_EXP_THRESHOLD, 100.0);
            T2 = vn_sel(m_ge, ex, min_exp);
            dT2_dVd = vn_sel(m_ge, -b->dvtp1 * ex, z0);
        }
        vnd T3 = b->leff + b->dvtp0 * (one + T2);
        vnd dT3_dVd = b->dvtp0 * dT2_dVd;
        vnd T4 = b->vtm * vn_log_pos(b->leff / T3);
        vnd dT4_dVd = -b->vtm * dT3_dVd / T3;
        vnd dDITS_dVd = dn_dVd * T4 + n * dT4_dVd;
        vnd dDITS_dVb = T4 * dn_dVb;
        Vth      = vn_sel(b->m_dvtp0, Vth - n * T4, Vth);
        dVth_dVd = vn_sel(b->m_dvtp0, dVth_dVd - dDITS_dVd, dVth_dVd);
        dVth_dVb = vn_sel(b->m_dvtp0, dVth_dVb - dDITS_dVb, dVth_dVb);
    }
    hot->von = Vth;
    if (need_aos)
        vn_store(out->von, Vth);

    /* ===================================================================== */
    /* 4. Poly Depletion & Vgsteff                                           */
    /* ===================================================================== */
    vnd vgs_eff, dvgs_eff_dvg, vgd_eff, dvgd_eff_dvg;
    vn_poly_depletion(b->vfb_plus_phi, b->polyT1, b->m_ngate_valid, vgs, &vgs_eff, &dvgs_eff_dvg);
    vn_poly_depletion(b->vfb_plus_phi, b->polyT1, b->m_ngate_valid, vgd, &vgd_eff, &dvgd_eff_dvg);
    vnd Vgs_eff = vn_sel(m_fwd, vgs_eff, vgd_eff);
    vnd dVgs_eff_dVg = vn_sel(m_fwd, dvgs_eff_dvg, dvgd_eff_dvg);

    vnd Vgst = Vgs_eff - Vth;

    vnd Vgsteff, dVgsteff_dVg, dVgsteff_dVd, dVgsteff_dVb;
    {
        vnd T0 = n * b->vtm;
        vnd T1 = b->mstar * Vgst;
        vnd T2 = T1 / T0;
        vnm m_hi = vn_gt(T2, exp_th);
        vnm m_lo = vn_lt(T2, nexp_th);
        vnm m_mid = (~m_hi) & (~m_lo);

        vnd T10_hi = T1;
        vnd dT10g_hi = b->mstar * dVgs_eff_dVg;
        vnd dT10d_hi = -dVth_dVd * b->mstar;
        vnd dT10b_hi = -dVth_dVb * b->mstar;

        vnd t10_base = b->vtm * vn_set1(B4V_LOG1P_MIN_EXP);
        vnd T10_lo = t10_base * n;
        vnd dT10d_lo = t10_base * dn_dVd;
        vnd dT10b_lo = t10_base * dn_dVb;

        vnd T10 = vn_sel(m_hi, T10_hi, T10_lo);
        vnd dT10g = vn_sel(m_hi, dT10g_hi, z0);
        vnd dT10d = vn_sel(m_hi, dT10d_hi, dT10d_lo);
        vnd dT10b = vn_sel(m_hi, dT10b_hi, dT10b_lo);
        if (vn_any(m_mid)) {
            vnd ex = vn_exp_clamped(T2, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            vnd T3 = b->vtm * vn_log_pos(one + ex);
            vnd dg0 = b->mstar * ex / (one + ex);
            vnd inv_n = one / n;
            T10   = vn_sel(m_mid, n * T3, T10);
            dT10b = vn_sel(m_mid, T3 * dn_dVb - dg0 * (dVth_dVb + Vgst * dn_dVb * inv_n), dT10b);
            dT10d = vn_sel(m_mid, T3 * dn_dVd - dg0 * (dVth_dVd + Vgst * dn_dVd * inv_n), dT10d);
            dT10g = vn_sel(m_mid, dg0 * dVgs_eff_dVg, dT10g);
        }

        vnd T1b = b->voffcbn - (one - b->mstar) * Vgst;
        vnd T2b = T1b / T0;
        vnm m2_lo = vn_lt(T2b, nexp_th);
        vnm m2_hi = vn_gt(T2b, exp_th);
        vnm m2_mid = (~m2_lo) & (~m2_hi);

        vnd T3_cl = b->coxe_ov_cdep0 * vn_sel(m2_lo, min_exp, max_exp);
        vnd T9 = b->mstar + T3_cl * n;
        vnd dT9g = z0;
        vnd dT9d = dn_dVd * T3_cl;
        vnd dT9b = dn_dVb * T3_cl;
        if (vn_any(m2_mid)) {
            vnd ex = vn_exp_clamped(T2b, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            vnd T4 = b->coxe_ov_cdep0 * ex;
            vnd T5 = T1b * T4 / T0;
            vnd dg0 = b->coxe_ov_cdep0 * (b->mstar - one) * ex / b->vtm;
            T9   = vn_sel(m2_mid, b->mstar + n * T4, T9);
            dT9b = vn_sel(m2_mid, T4 * dn_dVb - dg0 * dVth_dVb - T5 * dn_dVb, dT9b);
            dT9d = vn_sel(m2_mid, T4 * dn_dVd - dg0 * dVth_dVd - T5 * dn_dVd, dT9d);
            dT9g = vn_sel(m2_mid, dg0 * dVgs_eff_dVg, dT9g);
        }

        vnd invT9 = one / T9;
        vnd invT9sq = invT9 * invT9;
        Vgsteff      = T10 * invT9;
        dVgsteff_dVg = (T9 * dT10g - T10 * dT9g) * invT9sq;
        dVgsteff_dVd = (T9 * dT10d - T10 * dT9d) * invT9sq;
        dVgsteff_dVb = (T9 * dT10b - T10 * dT9b) * invT9sq;
    }

    /* ===================================================================== */
    /* 5. Weff, Rds (rdsMod == 0), Abulk                                     */
    /* ===================================================================== */
    vnd T9_phi = sqrtPhis - b->sqrtPhi;
    vnd Weff = b->weff - two * (b->dwg * Vgsteff + b->dwb * T9_phi);
    vnd dWeff_dVg = -two * b->dwg;
    vnd dWeff_dVb = -two * b->dwb * dsqrtPhis_dVb;
    {
        vnm m_w = vn_lt(Weff, vn_set1(2.0e-8));
        if (vn_any(m_w)) {
            vnd T0 = one / (vn_set1(6.0e-8) - two * Weff);
            vnd T0s = T0 * T0 * vn_set1(4.0e-16);
            Weff = vn_sel(m_w, vn_set1(2.0e-8) * (vn_set1(4.0e-8) - Weff) * T0, Weff);
            dWeff_dVg = vn_sel(m_w, dWeff_dVg * T0s, dWeff_dVg);
            dWeff_dVb = vn_sel(m_w, dWeff_dVb * T0s, dWeff_dVb);
        }
    }

    vnd Rds, dRds_dVg, dRds_dVb;
    {
        vnd T0 = one + b->prwg * Vgsteff;
        vnd invT0 = one / T0;
        vnd dT0_dVg = -b->prwg * invT0 * invT0;
        vnd T1 = b->prwb * T9_phi;
        vnd dT1_dVb = b->prwb * dsqrtPhis_dVb;
        vnd T2 = invT0 + T1;
        vnd T3 = T2 + vn_sqrt(T2 * T2 + vn_set1(0.01));
        vnd dT3_base = one + T2 / (T3 - T2);
        Rds = b->rdswmin + T3 * b->rds0_half;
        dRds_dVg = b->rds0_half * dT3_base * dT0_dVg;
        dRds_dVb = b->rds0_half * dT3_base * dT1_dVb;
    }

    vnd Abulk0, dAbulk0_dVb, Abulk, dAbulk_dVg, dAbulk_dVb;
    {
        vnd T9 = half * b->k1ox * b->lpebFactor / sqrtPhis;
        vnd T1 = T9 + b->k2ox - b->k3b * b->vthNarrowW;
        vnd dT1_dVb = -T9 / sqrtPhis * dsqrtPhis_dVb;

        vnd T9x = vn_sqrt(b->xj * Xdep);
        vnd tmp1 = b->leff + two * T9x;
        vnd T5 = b->leff / tmp1;
        vnd tmp2 = b->a0 * T5;
        vnd T2 = tmp2 + b->b0_ov_wb1;
        vnd dT2_dVb = -T9x / (tmp1 * Xdep) * dXdep_dVb;
        vnd T7 = T5 * T5 * T5;

        Abulk0 = one + T1 * T2;
        dAbulk0_dVb = T1 * tmp2 * dT2_dVb + T2 * dT1_dVb;

        vnd T8 = b->ags_a0 * T7;
        dAbulk_dVg = -T1 * T8;
        Abulk = Abulk0 + dAbulk_dVg * Vgsteff;
        dAbulk_dVb = dAbulk0_dVb - T8 * Vgsteff * (dT1_dVb + vn_set1(3.0) * T1 * dT2_dVb);

        vnm m_a0 = vn_lt(Abulk0, vn_set1(0.1));
        if (vn_any(m_a0)) {
            vnd Ta = one / (vn_set1(3.0) - vn_set1(20.0) * Abulk0);
            Abulk0 = vn_sel(m_a0, (vn_set1(0.2) - Abulk0) * Ta, Abulk0);
            dAbulk0_dVb = vn_sel(m_a0, dAbulk0_dVb * Ta * Ta, dAbulk0_dVb);
        }
        vnm m_ab = vn_lt(Abulk, vn_set1(0.1));
        if (vn_any(m_ab)) {
            vnd Ta = one / (vn_set1(3.0) - vn_set1(20.0) * Abulk);
            vnd Tasq = Ta * Ta;
            Abulk = vn_sel(m_ab, (vn_set1(0.2) - Abulk) * Ta, Abulk);
            dAbulk_dVb = vn_sel(m_ab, dAbulk_dVb * Tasq, dAbulk_dVb);
            dAbulk_dVg = vn_sel(m_ab, dAbulk_dVg * Tasq, dAbulk_dVg);
        }

        vnd Tk = b->keta * Vbseff;
        vnm m_k = vn_ge(Tk, vn_set1(-0.9));
        vnd T0k = one / (one + Tk);
        vnd T1k = one / (vn_set1(0.8) + Tk);
        vnd T0 = vn_sel(m_k, T0k, (vn_set1(17.0) + vn_set1(20.0) * Tk) * T1k);
        vnd dT0_dVb = vn_sel(m_k, -b->keta * T0k * T0k, -b->keta * T1k * T1k);
        dAbulk_dVg = dAbulk_dVg * T0;
        dAbulk_dVb = dAbulk_dVb * T0 + Abulk * dT0_dVb;
        dAbulk0_dVb = dAbulk0_dVb * T0 + Abulk0 * dT0_dVb;
        Abulk = Abulk * T0;
        Abulk0 = Abulk0 * T0;
    }

    /* ===================================================================== */
    /* 6. Mobility (mobMod == 0), EsatL, Vdsat, Vdseff                       */
    /* ===================================================================== */
    vnd ueff, dueff_dVg, dueff_dVd, dueff_dVb;
    {
        vnd T0 = Vgsteff + Vth + Vth;
        vnd T2 = b->ua + b->uc * Vbseff;
        vnd T3 = T0 * b->invToxe;
        vnd T5, dDen_dVg, dDen_dVd, dDen_dVb;
        if (!b->has_ud) {
            T5 = T3 * (T2 + b->ub * T3);
            dDen_dVg = (T2 + two * b->ub * T3) * b->invToxe;
            dDen_dVd = dDen_dVg * two * dVth_dVd;
            dDen_dVb = dDen_dVg * two * dVth_dVb + b->uc * T3;
        } else if (!b->is_v6) {
            vnd invT3 = one / T3;
            vnd T6 = b->ud * invT3 * invT3 * Vth * Vth;
            T5 = T3 * (T2 + b->ub * T3) + T6;
            vnd T7 = -two * T6 / T0;
            dDen_dVg = (T2 + two * b->ub * T3) * b->invToxe + T7;
            dDen_dVd = dDen_dVg * two * dVth_dVd;
            dDen_dVb = dDen_dVg * two * dVth_dVb + b->uc * T3;
        } else {
            vnd T12 = vn_sqrt(Vth * Vth + vn_set1(0.0001));
            vnd T9 = one / (Vgsteff + two * T12);
            vnd T10 = T9 * b->toxe;
            vnd T8 = b->ud * T10 * T10 * Vth;
            vnd T6 = T8 * Vth;
            T5 = T3 * (T2 + b->ub * T3) + T6;
            vnd T7 = -two * T6 * T9;
            vnd T11 = T7 * Vth / T12;
            vnd base_g = (T2 + two * b->ub * T3) * b->invToxe;
            vnd T13 = two * (base_g + T11 + T8);
            dDen_dVd = T13 * dVth_dVd;
            dDen_dVb = T13 * dVth_dVb + b->uc * T3;
            dDen_dVg = base_g + T7;
        }

        vnm m_den = vn_ge(T5, vn_set1(-0.8));
        vnd T9 = one / (vn_set1(7.0) + vn_set1(10.0) * T5);
        vnd Denomi = vn_sel(m_den, one + T5, (vn_set1(0.6) + T5) * T9);
        vnd T9sq = T9 * T9;
        dDen_dVg = vn_sel(m_den, dDen_dVg, dDen_dVg * T9sq);
        dDen_dVd = vn_sel(m_den, dDen_dVd, dDen_dVd * T9sq);
        dDen_dVb = vn_sel(m_den, dDen_dVb, dDen_dVb * T9sq);

        vnd invDen = one / Denomi;
        ueff = b->u0temp * invDen;
        vnd m_u_inv = -ueff * invDen;
        dueff_dVg = m_u_inv * dDen_dVg;
        dueff_dVd = m_u_inv * dDen_dVd;
        dueff_dVb = m_u_inv * dDen_dVb;
    }

    vnd WVCox = Weff * b->vsattemp * b->coxe;
    vnd WVCoxRds = WVCox * Rds;

    vnd Esat = two * b->vsattemp / ueff;
    vnd EsatL = Esat * b->leff;
    vnd T0_es = -EsatL / ueff;
    vnd dEsatL_dVg = T0_es * dueff_dVg;
    vnd dEsatL_dVd = T0_es * dueff_dVd;
    vnd dEsatL_dVb = T0_es * dueff_dVb;

    vnd Lambda = b->a2, dLambda_dVg = z0;
    if (b->f_has_a1) {
        vnm m_pos = vn_gt(b->a1, z0);
        vnm m_neg = vn_lt(b->a1, z0);
        if (vn_any(m_pos)) {
            vnd T0 = one - b->a2;
            vnd T1 = T0 - b->a1 * Vgsteff - vn_set1(0.0001);
            vnd T2 = vn_sqrt(vn_max(T1 * T1 + vn_set1(0.0004) * T0, vn_set1(1.0e-300)));
            Lambda = vn_sel(m_pos, b->a2 + T0 - half * (T1 + T2), Lambda);
            dLambda_dVg = vn_sel(m_pos, half * b->a1 * (one + T1 / T2), dLambda_dVg);
        }
        if (vn_any(m_neg)) {
            vnd T1 = b->a2 + b->a1 * Vgsteff - vn_set1(0.0001);
            vnd T2 = vn_sqrt(vn_max(T1 * T1 + vn_set1(0.0004) * b->a2, vn_set1(1.0e-300)));
            Lambda = vn_sel(m_neg, half * (T1 + T2), Lambda);
            dLambda_dVg = vn_sel(m_neg, half * b->a1 * (one + T1 / T2), dLambda_dVg);
        }
    }

    vnd Vgst2Vtm = Vgsteff + two * b->vtm;
    vnd invWeff = one / Weff;
    vnm m_rds_pos = vn_gt(Rds, z0);
    vnd invRds_safe = vn_sel(m_rds_pos, one / vn_max(Rds, vn_set1(1.0e-300)), z0);
    vnd tmp2_rw = dRds_dVg * invRds_safe + dWeff_dVg * invWeff;
    vnd tmp3_rw = dRds_dVb * invRds_safe + dWeff_dVb * invWeff;

    vnd Vdsat, dVdsat_dVg, dVdsat_dVd, dVdsat_dVb, tmp1_lam;
    {
        vnm m_simple = vn_eq(Rds, z0) & vn_eq(Lambda, one);
        if (vn_all(m_simple)) {
            vnd T0 = one / (Abulk * EsatL + Vgst2Vtm);
            tmp1_lam = z0;
            vnd T1 = T0 * T0;
            vnd T2 = Vgst2Vtm * T0;
            vnd T3 = EsatL * Vgst2Vtm;
            Vdsat = T3 * T0;
            vnd dT0_dVg = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + one) * T1;
            vnd dT0_dVd = -(Abulk * dEsatL_dVd) * T1;
            vnd dT0_dVb = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * T1;
            dVdsat_dVg = T3 * dT0_dVg + T2 * dEsatL_dVg + EsatL * T0;
            dVdsat_dVd = T3 * dT0_dVd + T2 * dEsatL_dVd;
            dVdsat_dVb = T3 * dT0_dVb + T2 * dEsatL_dVb;
        } else {
            vnd invLam = one / Lambda;
            vnd twoInvLamM1 = two * invLam - one;
            vnd invLamM1 = invLam - one;
            tmp1_lam = dLambda_dVg * invLam * invLam;
            vnd T9 = Abulk * WVCoxRds;
            vnd T8 = Abulk * T9;
            vnd T7 = Vgst2Vtm * T9;
            vnd T6 = Vgst2Vtm * WVCoxRds;
            vnd T0 = two * Abulk * (T9 - one + invLam);
            vnd dT0_dVg = two * (T8 * tmp2_rw - Abulk * tmp1_lam + (two * T9 + invLamM1) * dAbulk_dVg);
            vnd dT0_dVb = two * (T8 * (two / Abulk * dAbulk_dVb + tmp3_rw) + invLamM1 * dAbulk_dVb);
            vnd T1 = Vgst2Vtm * twoInvLamM1 + Abulk * EsatL + vn_set1(3.0) * T7;
            vnd dT1_dVg = twoInvLamM1 - two * Vgst2Vtm * tmp1_lam
                        + Abulk * dEsatL_dVg + EsatL * dAbulk_dVg
                        + vn_set1(3.0) * (T9 + T7 * tmp2_rw + T6 * dAbulk_dVg);
            vnd dT1_dVb = Abulk * dEsatL_dVb + EsatL * dAbulk_dVb
                        + vn_set1(3.0) * (T6 * dAbulk_dVb + T7 * tmp3_rw);
            vnd dT1_dVd = Abulk * dEsatL_dVd;
            vnd T2 = Vgst2Vtm * (EsatL + two * T6);
            vnd dT2_dVg = EsatL + Vgst2Vtm * dEsatL_dVg + T6 * (vn_set1(4.0) + two * Vgst2Vtm * tmp2_rw);
            vnd dT2_dVb = Vgst2Vtm * (dEsatL_dVb + two * T6 * tmp3_rw);
            vnd dT2_dVd = Vgst2Vtm * dEsatL_dVd;
            vnd T3 = vn_sqrt(vn_max(T1 * T1 - two * T0 * T2, vn_set1(1.0e-300)));
            vnd invT0 = one / vn_sel(m_simple, one, T0);
            vnd invT3 = one / T3;
            Vdsat = (T1 - T3) * invT0;
            dVdsat_dVg = (dT1_dVg - (T1 * dT1_dVg - dT0_dVg * T2 - T0 * dT2_dVg) * invT3 - Vdsat * dT0_dVg) * invT0;
            dVdsat_dVb = (dT1_dVb - (T1 * dT1_dVb - dT0_dVb * T2 - T0 * dT2_dVb) * invT3 - Vdsat * dT0_dVb) * invT0;
            dVdsat_dVd = (dT1_dVd - (T1 * dT1_dVd - T0 * dT2_dVd) * invT3) * invT0;
            if (vn_any(m_simple)) {
                vnd sT0 = one / (Abulk * EsatL + Vgst2Vtm);
                vnd sT1 = sT0 * sT0;
                vnd sT2 = Vgst2Vtm * sT0;
                vnd sT3 = EsatL * Vgst2Vtm;
                vnd sdT0g = -(Abulk * dEsatL_dVg + EsatL * dAbulk_dVg + one) * sT1;
                vnd sdT0d = -(Abulk * dEsatL_dVd) * sT1;
                vnd sdT0b = -(Abulk * dEsatL_dVb + dAbulk_dVb * EsatL) * sT1;
                Vdsat      = vn_sel(m_simple, sT3 * sT0, Vdsat);
                dVdsat_dVg = vn_sel(m_simple, sT3 * sdT0g + sT2 * dEsatL_dVg + EsatL * sT0, dVdsat_dVg);
                dVdsat_dVd = vn_sel(m_simple, sT3 * sdT0d + sT2 * dEsatL_dVd, dVdsat_dVd);
                dVdsat_dVb = vn_sel(m_simple, sT3 * sdT0b + sT2 * dEsatL_dVb, dVdsat_dVb);
                tmp1_lam   = vn_sel(m_simple, z0, tmp1_lam);
            }
        }
    }

    /* Vdseff */
    vnd Vdseff, dVdseff_dVg, dVdseff_dVd, dVdseff_dVb;
    {
        vnd T1 = Vdsat - Vds - b->delta;
        vnd dT1_dVg = dVdsat_dVg;
        vnd dT1_dVd = dVdsat_dVd - one;
        vnd dT1_dVb = dVdsat_dVb;
        vnd T2 = vn_sqrt(T1 * T1 + vn_set1(4.0) * b->delta * Vdsat);
        vnd invT2 = one / T2;
        vnd T0 = T1 * invT2;
        vnd T9 = two * b->delta;
        vnd T3 = T9 * invT2;
        vnd dT2_dVg = T0 * dT1_dVg + T3 * dVdsat_dVg;
        vnd dT2_dVd = T0 * dT1_dVd + T3 * dVdsat_dVd;
        vnd dT2_dVb = T0 * dT1_dVb + T3 * dVdsat_dVb;

        vnm m_ge = vn_ge(T1, z0);
        vnd ve_p = Vdsat - half * (T1 + T2);
        vnd dg_p = dVdsat_dVg - half * (dT1_dVg + dT2_dVg);
        vnd dd_p = dVdsat_dVd - half * (dT1_dVd + dT2_dVd);
        vnd db_p = dVdsat_dVb - half * (dT1_dVb + dT2_dVb);

        vnd inv_diff = one / (T2 - T1);
        vnd T4 = T9 * inv_diff;
        vnd T5 = one - T4;
        vnd T6 = Vdsat * T4 * inv_diff;
        vnd ve_n = Vdsat * T5;
        vnd dg_n = dVdsat_dVg * T5 + T6 * (dT2_dVg - dT1_dVg);
        vnd dd_n = dVdsat_dVd * T5 + T6 * (dT2_dVd - dT1_dVd);
        vnd db_n = dVdsat_dVb * T5 + T6 * (dT2_dVb - dT1_dVb);

        Vdseff      = vn_sel(m_ge, ve_p, ve_n);
        dVdseff_dVg = vn_sel(m_ge, dg_p, dg_n);
        dVdseff_dVd = vn_sel(m_ge, dd_p, dd_n);
        dVdseff_dVb = vn_sel(m_ge, db_p, db_n);

        vnm m_vds0 = vn_eq(Vds, z0);
        Vdseff      = vn_sel(m_vds0, z0, Vdseff);
        dVdseff_dVg = vn_sel(m_vds0, z0, dVdseff_dVg);
        dVdseff_dVb = vn_sel(m_vds0, z0, dVdseff_dVb);
        Vdseff      = vn_min(Vdseff, Vds);
    }
    vnd diffVds = Vds - Vdseff;

    /* Velocity Overshoot (lambda > 0) */
    if (b->f_lambda) {
        vnd T1 = b->leff * ueff;
        vnd T2 = b->lambda / T1;
        vnd T3 = -T2 / T1 * b->leff;
        vnd dT2_dVd = T3 * dueff_dVd;
        vnd dT2_dVg = T3 * dueff_dVg;
        vnd dT2_dVb = T3 * dueff_dVb;
        vnd T5 = one / (Esat * b->litl);
        vnd T4 = -T5 / EsatL;
        vnd dT5_dVg = dEsatL_dVg * T4;
        vnd dT5_dVd = dEsatL_dVd * T4;
        vnd dT5_dVb = dEsatL_dVb * T4;
        vnd T6 = one + diffVds * T5;
        vnd dT6_dVg = dT5_dVg * diffVds - dVdseff_dVg * T5;
        vnd dT6_dVd = dT5_dVd * diffVds + (one - dVdseff_dVd) * T5;
        vnd dT6_dVb = dT5_dVb * diffVds - dVdseff_dVb * T5;
        vnd T7 = two / (T6 * T6 + one);
        vnd T8 = one - T7;
        vnd T9 = T6 * T7 * T7;
        vnd T10 = one + T2 * T8;
        vnm m_eq1 = vn_eq(T10, one);
        vnd dT10_dVg = vn_sel(m_eq1, z0, dT2_dVg * T8 + T2 * T9 * dT6_dVg);
        vnd dT10_dVd = vn_sel(m_eq1, z0, dT2_dVd * T8 + T2 * T9 * dT6_dVd);
        vnd dT10_dVb = vn_sel(m_eq1, z0, dT2_dVb * T8 + T2 * T9 * dT6_dVb);
        dEsatL_dVg = vn_sel(b->m_lambda, dEsatL_dVg * T10 + EsatL * dT10_dVg, dEsatL_dVg);
        dEsatL_dVd = vn_sel(b->m_lambda, dEsatL_dVd * T10 + EsatL * dT10_dVd, dEsatL_dVd);
        dEsatL_dVb = vn_sel(b->m_lambda, dEsatL_dVb * T10 + EsatL * dT10_dVb, dEsatL_dVb);
        EsatL      = vn_sel(b->m_lambda, EsatL * T10, EsatL);
        if (b->is_v6)
            Esat = vn_sel(b->m_lambda, EsatL * b->invLeff, Esat);
    }

    /* ===================================================================== */
    /* 7. Vasat, Idl, CLM, DIBL, DITS, SCBE, Isub, Ids, Fsevl                */
    /* ===================================================================== */
    vnd Vasat, dVasat_dVg, dVasat_dVb, dVasat_dVd;
    {
        vnd invVgst2Vtm = one / Vgst2Vtm;
        vnd tmp4 = one - half * Abulk * Vdsat * invVgst2Vtm;
        vnd T9 = WVCoxRds * Vgsteff;
        vnd T8 = T9 * invVgst2Vtm;
        vnd T0 = EsatL + Vdsat + two * T9 * tmp4;
        vnd T7 = two * WVCoxRds * tmp4;
        vnd dT0_dVg = dEsatL_dVg + dVdsat_dVg + T7 * (one + tmp2_rw * Vgsteff)
                    - T8 * (Abulk * dVdsat_dVg - Abulk * Vdsat * invVgst2Vtm + Vdsat * dAbulk_dVg);
        vnd dT0_dVb = dEsatL_dVb + dVdsat_dVb + T7 * tmp3_rw * Vgsteff
                    - T8 * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
        vnd dT0_dVd = dEsatL_dVd + dVdsat_dVd - T8 * Abulk * dVdsat_dVd;
        vnd T9b = WVCoxRds * Abulk;
        vnd T1 = two / Lambda - one + T9b;
        vnd dT1_dVg = -two * tmp1_lam + WVCoxRds * (Abulk * tmp2_rw + dAbulk_dVg);
        vnd dT1_dVb = dAbulk_dVb * WVCoxRds + T9b * tmp3_rw;
        vnd invT1 = one / T1;
        Vasat = T0 * invT1;
        dVasat_dVg = (dT0_dVg - Vasat * dT1_dVg) * invT1;
        dVasat_dVb = (dT0_dVb - Vasat * dT1_dVb) * invT1;
        dVasat_dVd = dT0_dVd * invT1;
    }

    /* Idl & CoxeffWovL */
    vnd CoxeffWovL, Idl, dIdl_dVg, dIdl_dVd, dIdl_dVb;
    {
        vnd T0 = (Vgsteff + b->vtfbphi2) * b->inv_2e8_toxp;
        vnd tmp3 = vn_exp(b->bdos0p7 * vn_log_pos(T0));
        vnd T1 = one + tmp3;
        vnd invT1 = one / T1;
        vnd T2 = b->bdos0p7 * tmp3 / T0;
        vnd Tcen = b->ados1p9 * invT1;
        vnd dTcen_dVg = -Tcen * T2 * b->inv_2e8_toxp * invT1;
        vnd Coxeff = epssi * b->coxp / (epssi + b->coxp * Tcen);
        vnd dCoxeff_dVg = -Coxeff * Coxeff * dTcen_dVg * vn_set1(1.0 / B4V_EPSSI);
        CoxeffWovL = Coxeff * Weff * b->invLeff;
        vnd beta = ueff * CoxeffWovL;
        vnd T3 = ueff * b->invLeff;
        vnd dbeta_dVg = CoxeffWovL * dueff_dVg + T3 * (Weff * dCoxeff_dVg + Coxeff * dWeff_dVg);
        vnd dbeta_dVd = CoxeffWovL * dueff_dVd;
        vnd dbeta_dVb = CoxeffWovL * dueff_dVb + T3 * Coxeff * dWeff_dVb;

        vnd invVgst2Vtm = one / Vgst2Vtm;
        vnd T0a = one - half * Vdseff * Abulk * invVgst2Vtm;
        vnd dT0a_dVg = -half * (Abulk * dVdseff_dVg - Abulk * Vdseff * invVgst2Vtm + Vdseff * dAbulk_dVg) * invVgst2Vtm;
        vnd dT0a_dVd = -half * Abulk * dVdseff_dVd * invVgst2Vtm;
        vnd dT0a_dVb = -half * (Abulk * dVdseff_dVb + dAbulk_dVb * Vdseff) * invVgst2Vtm;

        vnd fgche1 = Vgsteff * T0a;
        vnd dfgche1_dVg = Vgsteff * dT0a_dVg + T0a;
        vnd dfgche1_dVd = Vgsteff * dT0a_dVd;
        vnd dfgche1_dVb = Vgsteff * dT0a_dVb;

        vnd invEsatL = one / EsatL;
        vnd T9 = Vdseff * invEsatL;
        vnd fgche2 = one + T9;
        vnd invFg2 = one / fgche2;
        vnd dfgche2_dVg = (dVdseff_dVg - T9 * dEsatL_dVg) * invEsatL;
        vnd dfgche2_dVd = (dVdseff_dVd - T9 * dEsatL_dVd) * invEsatL;
        vnd dfgche2_dVb = (dVdseff_dVb - T9 * dEsatL_dVb) * invEsatL;

        vnd gche = beta * fgche1 * invFg2;
        vnd dgche_dVg = (beta * dfgche1_dVg + fgche1 * dbeta_dVg - gche * dfgche2_dVg) * invFg2;
        vnd dgche_dVd = (beta * dfgche1_dVd + fgche1 * dbeta_dVd - gche * dfgche2_dVd) * invFg2;
        vnd dgche_dVb = (beta * dfgche1_dVb + fgche1 * dbeta_dVb - gche * dfgche2_dVb) * invFg2;

        vnd T0g = one + gche * Rds;
        vnd invT0g = one / T0g;
        Idl = gche * invT0g;
        vnd T1g = (one - Idl * Rds) * invT0g;
        vnd T2g = Idl * Idl;
        dIdl_dVg = T1g * dgche_dVg - T2g * dRds_dVg;
        dIdl_dVd = T1g * dgche_dVd;
        dIdl_dVb = T1g * dgche_dVb - T2g * dRds_dVb;
    }

    /* FP (pocket degradation) */
    vnd FP = one, dFP_dVg = z0;
    if (b->f_fprout) {
        vnd T9 = b->fprout * b->sqrtLeff / Vgst2Vtm;
        vnd fp = one / (one + T9);
        FP = vn_sel(b->m_fprout, fp, one);
        dFP_dVg = vn_sel(b->m_fprout, fp * fp * T9 / Vgst2Vtm, z0);
    }

    /* PvagTerm */
    vnd PvagTerm, dPvagTerm_dVg, dPvagTerm_dVb, dPvagTerm_dVd;
    {
        vnd invEsatL = one / EsatL;
        vnd T8 = b->pvag * invEsatL;
        vnd T9 = T8 * Vgsteff;
        vnm m_gt = vn_gt(T9, vn_set1(-0.9));
        vnd T4 = one / (vn_set1(17.0) + vn_set1(20.0) * T9);
        vnd T4sq = T4 * T4;
        vnd scale_d = vn_sel(m_gt, one, T4sq);
        PvagTerm = vn_sel(m_gt, one + T9, (vn_set1(0.8) + T9) * T4);
        dPvagTerm_dVg = T8 * (one - Vgsteff * dEsatL_dVg * invEsatL) * scale_d;
        vnd f_bd = -T9 * invEsatL * scale_d;
        dPvagTerm_dVb = f_bd * dEsatL_dVb;
        dPvagTerm_dVd = f_bd * dEsatL_dVd;
    }

    /* VACLM & Cclm */
    vnd Cclm = max_exp, dCclm_dVg = z0, dCclm_dVb = z0, dCclm_dVd = z0;
    vnd VACLM = max_exp, dVACLM_dVg = z0, dVACLM_dVb = z0, dVACLM_dVd = z0;
    {
        vnm m_clm = b->m_pclm & vn_gt(diffVds, vn_set1(1.0e-10));
        if (vn_any(m_clm)) {
            vnd T0 = one + Rds * Idl;
            vnd dT0_dVg = dRds_dVg * Idl + Rds * dIdl_dVg;
            vnd dT0_dVd = Rds * dIdl_dVd;
            vnd dT0_dVb = dRds_dVb * Idl + Rds * dIdl_dVb;
            vnd invEsat = one / Esat;
            vnd T2 = Vdsat * invEsat;
            vnd T1 = b->leff + T2;
            vnd dT1_dVg = (dVdsat_dVg - T2 * dEsatL_dVg * b->invLeff) * invEsat;
            vnd dT1_dVd = (dVdsat_dVd - T2 * dEsatL_dVd * b->invLeff) * invEsat;
            vnd dT1_dVb = (dVdsat_dVb - T2 * dEsatL_dVb * b->invLeff) * invEsat;
            vnd cc = FP * PvagTerm * T0 * T1 * b->inv_pclm_litl;
            vnd invPv = one / PvagTerm, invT0 = one / T0, invT1 = one / T1;
            vnd dcc_g = cc * (dFP_dVg / FP + dPvagTerm_dVg * invPv + dT0_dVg * invT0 + dT1_dVg * invT1);
            vnd dcc_b = cc * (dPvagTerm_dVb * invPv + dT0_dVb * invT0 + dT1_dVb * invT1);
            vnd dcc_d = cc * (dPvagTerm_dVd * invPv + dT0_dVd * invT0 + dT1_dVd * invT1);
            Cclm = vn_sel(m_clm, cc, max_exp);
            dCclm_dVg = vn_sel(m_clm, dcc_g, z0);
            dCclm_dVb = vn_sel(m_clm, dcc_b, z0);
            dCclm_dVd = vn_sel(m_clm, dcc_d, z0);
            VACLM = vn_sel(m_clm, cc * diffVds, max_exp);
            dVACLM_dVg = vn_sel(m_clm, dcc_g * diffVds - dVdseff_dVg * cc, z0);
            dVACLM_dVb = vn_sel(m_clm, dcc_b * diffVds - dVdseff_dVb * cc, z0);
            dVACLM_dVd = vn_sel(m_clm, dcc_d * diffVds + (one - dVdseff_dVd) * cc, z0);
        }
    }

    /* VADIBL */
    vnd VADIBL = max_exp, dVADIBL_dVg = z0, dVADIBL_dVb = z0, dVADIBL_dVd = z0;
    if (b->f_thetaRout) {
        vnd T8 = Abulk * Vdsat;
        vnd T0 = Vgst2Vtm * T8;
        vnd dT0_dVg = Vgst2Vtm * Abulk * dVdsat_dVg + T8 + Vgst2Vtm * Vdsat * dAbulk_dVg;
        vnd dT0_dVb = Vgst2Vtm * (dAbulk_dVb * Vdsat + Abulk * dVdsat_dVb);
        vnd dT0_dVd = Vgst2Vtm * Abulk * dVdsat_dVd;
        vnd T1 = Vgst2Vtm + T8;
        vnd dT1_dVg = one + Abulk * dVdsat_dVg + Vdsat * dAbulk_dVg;
        vnd dT1_dVb = Abulk * dVdsat_dVb + dAbulk_dVb * Vdsat;
        vnd dT1_dVd = Abulk * dVdsat_dVd;
        vnd invT1 = one / T1;
        vnd invT1sq = invT1 * invT1;
        vnd va = (Vgst2Vtm - T0 * invT1) * b->inv_thetaRout;
        vnd dva_g = (one - dT0_dVg * invT1 + T0 * dT1_dVg * invT1sq) * b->inv_thetaRout;
        vnd dva_b = (-dT0_dVb * invT1 + T0 * dT1_dVb * invT1sq) * b->inv_thetaRout;
        vnd dva_d = (-dT0_dVd * invT1 + T0 * dT1_dVd * invT1sq) * b->inv_thetaRout;

        vnd T7 = b->pdiblb * Vbseff;
        vnm m_ge = vn_ge(T7, vn_set1(-0.9));
        vnd T3_p = one / (one + T7);
        vnd T4_n = one / (vn_set1(0.8) + T7);
        vnd T3_n = (vn_set1(17.0) + vn_set1(20.0) * T7) * T4_n;
        vnd T3 = vn_sel(m_ge, T3_p, T3_n);
        vnd va_p = va * T3_p;
        vnd dvab_p = (dva_b - va_p * b->pdiblb) * T3_p;
        vnd dvab_n = dva_b * T3_n - va * b->pdiblb * T4_n * T4_n;
        va = va * T3;
        dva_g = dva_g * T3;
        dva_d = dva_d * T3;
        dva_b = vn_sel(m_ge, dvab_p, dvab_n);

        dva_g = dva_g * PvagTerm + va * dPvagTerm_dVg;
        dva_b = dva_b * PvagTerm + va * dPvagTerm_dVb;
        dva_d = dva_d * PvagTerm + va * dPvagTerm_dVd;
        va = va * PvagTerm;

        VADIBL      = vn_sel(b->m_thetaRout, va, max_exp);
        dVADIBL_dVg = vn_sel(b->m_thetaRout, dva_g, z0);
        dVADIBL_dVb = vn_sel(b->m_thetaRout, dva_b, z0);
        dVADIBL_dVd = vn_sel(b->m_thetaRout, dva_d, z0);
    }

    vnd Va = Vasat + VACLM;
    vnd dVa_dVg = dVasat_dVg + dVACLM_dVg;
    vnd dVa_dVb = dVasat_dVb + dVACLM_dVb;
    vnd dVa_dVd = dVasat_dVd + dVACLM_dVd;

    /* VADITS (only compute exp when pdits > MIN_EXP!) */
    vnd VADITS = max_exp, dVADITS_dVg = z0, dVADITS_dVd = z0;
    if (b->f_pdits) {
        vnd T0 = b->pditsd * Vds;
        vnm m_le = vn_le(T0, exp_th) & b->m_pdits;
        vnd T1 = max_exp, dT1_dVd = z0;
        if (vn_any(m_le)) {
            vnd ex = vn_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD);
            T1 = vn_sel(m_le, ex, max_exp);
            dT1_dVd = vn_sel(m_le, ex * b->pditsd, z0);
        }
        vnd vd0 = (one + b->pditsl_term * T1) * b->inv_pdits;
        VADITS      = vn_sel(b->m_pdits, vd0 * FP, max_exp);
        dVADITS_dVg = vn_sel(b->m_pdits, vd0 * dFP_dVg, z0);
        dVADITS_dVd = vn_sel(b->m_pdits, FP * b->pditsl_term * dT1_dVd * b->inv_pdits, z0);
    }

    /* VASCBE */
    vnd VASCBE = max_exp, dVASCBE_dVg = z0, dVASCBE_dVd = z0, dVASCBE_dVb = z0;
    if (b->f_pscbe) {
        vnd scbe_hi = max_exp * b->leff * b->inv_pscbe2;
        vnd thr = b->pscbe1_litl * vn_set1(1.0 / B4V_EXP_THRESHOLD);
        vnm m_act = b->m_pscbe & vn_gt(diffVds, thr);
        VASCBE = vn_sel(b->m_pscbe, scbe_hi, max_exp);
        if (vn_any(m_act)) {
            vnd diff_safe = vn_max(diffVds, vn_set1(1.0e-30));
            vnd T0 = b->pscbe1_litl / diff_safe;
            vnd vs = b->leff * vn_exp_clamped(T0, -100.0, B4V_EXP_THRESHOLD) * b->inv_pscbe2;
            vnd T1 = T0 * vs / diff_safe;
            VASCBE      = vn_sel(m_act, vs, VASCBE);
            dVASCBE_dVg = vn_sel(m_act, T1 * dVdseff_dVg, z0);
            dVASCBE_dVd = vn_sel(m_act, -T1 * (one - dVdseff_dVd), z0);
            dVASCBE_dVb = vn_sel(m_act, T1 * dVdseff_dVb, z0);
        }
    }

    /* Add DIBL */
    vnd invVADIBL = one / VADIBL;
    vnd T9_dibl = diffVds * invVADIBL;
    vnd T0_dibl = one + T9_dibl;
    vnd Idsa = Idl * T0_dibl;
    vnd dIdsa_dVg = T0_dibl * dIdl_dVg - Idl * (dVdseff_dVg + T9_dibl * dVADIBL_dVg) * invVADIBL;
    vnd dIdsa_dVd = T0_dibl * dIdl_dVd + Idl * (one - dVdseff_dVd - T9_dibl * dVADIBL_dVd) * invVADIBL;
    vnd dIdsa_dVb = T0_dibl * dIdl_dVb - Idl * (dVdseff_dVb + T9_dibl * dVADIBL_dVb) * invVADIBL;

    /* Add DITS */
    {
        vnd invVADITS = one / VADITS;
        vnd T9 = diffVds * invVADITS;
        vnd T0 = one + T9;
        dIdsa_dVg = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVADITS_dVg) * invVADITS;
        dIdsa_dVd = T0 * dIdsa_dVd + Idsa * (one - dVdseff_dVd - T9 * dVADITS_dVd) * invVADITS;
        dIdsa_dVb = T0 * dIdsa_dVb - Idsa * dVdseff_dVb * invVADITS;
        Idsa = Idsa * T0;
    }

    /* Add CLM */
    {
        vnd T0 = vn_log_pos(Va / Vasat);
        vnd invVa = one / Va, invVasat = one / Vasat, invCclm = one / Cclm;
        vnd dT0_dVg = dVa_dVg * invVa - dVasat_dVg * invVasat;
        vnd dT0_dVb = dVa_dVb * invVa - dVasat_dVb * invVasat;
        vnd dT0_dVd = dVa_dVd * invVa - dVasat_dVd * invVasat;
        vnd T1 = T0 * invCclm;
        vnd T9 = one + T1;
        vnd dT9_dVg = (dT0_dVg - T1 * dCclm_dVg) * invCclm;
        vnd dT9_dVb = (dT0_dVb - T1 * dCclm_dVb) * invCclm;
        vnd dT9_dVd = (dT0_dVd - T1 * dCclm_dVd) * invCclm;
        dIdsa_dVg = dIdsa_dVg * T9 + Idsa * dT9_dVg;
        dIdsa_dVb = dIdsa_dVb * T9 + Idsa * dT9_dVb;
        dIdsa_dVd = dIdsa_dVd * T9 + Idsa * dT9_dVd;
        Idsa = Idsa * T9;
    }

    /* Substrate current Isub */
    vnd Isub = z0, Gbg = z0, Gbd = z0, Gbb = z0;
    if (b->f_isub) {
        vnd thr = b->beta0 * vn_set1(1.0 / B4V_EXP_THRESHOLD);
        vnm m_exp = b->m_isub & vn_gt(diffVds, thr);
        vnd T3_lo = b->alpha_ov_leff * min_exp;
        vnd T1 = T3_lo * diffVds;
        vnd dT1_dVg = -T3_lo * dVdseff_dVg;
        vnd dT1_dVd = T3_lo * (one - dVdseff_dVd);
        vnd dT1_dVb = -T3_lo * dVdseff_dVb;
        if (vn_any(m_exp)) {
            vnd diff_safe = vn_max(diffVds, vn_set1(1.0e-30));
            vnd T0 = -b->beta0 / diff_safe;
            vnd T1e = b->alpha_ov_leff * diff_safe * vn_exp_clamped(T0, -B4V_EXP_THRESHOLD, 0.0);
            vnd T3e = T1e / diff_safe * (T0 - one);
            T1      = vn_sel(m_exp, T1e, T1);
            dT1_dVg = vn_sel(m_exp, T3e * dVdseff_dVg, dT1_dVg);
            dT1_dVd = vn_sel(m_exp, T3e * (dVdseff_dVd - one), dT1_dVd);
            dT1_dVb = vn_sel(m_exp, T3e * dVdseff_dVb, dT1_dVb);
        }
        vnd T4 = Idsa * Vdseff;
        vnd gbg0 = T1 * (dIdsa_dVg * Vdseff + Idsa * dVdseff_dVg) + T4 * dT1_dVg;
        vnd gbd0 = T1 * (dIdsa_dVd * Vdseff + Idsa * dVdseff_dVd) + T4 * dT1_dVd;
        vnd gbb0 = T1 * (dIdsa_dVb * Vdseff + Idsa * dVdseff_dVb) + T4 * dT1_dVb;
        gbd0 = gbd0 + gbg0 * dVgsteff_dVd;
        gbb0 = (gbb0 + gbg0 * dVgsteff_dVb) * dVbseff_dVb;
        gbg0 = gbg0 * dVgsteff_dVg;
        Isub = vn_sel(b->m_isub, T1 * T4, z0);
        Gbg  = vn_sel(b->m_isub, gbg0, z0);
        Gbd  = vn_sel(b->m_isub, gbd0, z0);
        Gbb  = vn_sel(b->m_isub, gbb0, z0);
    }

    /* Add SCBE & assemble Ids, Gm, Gds, Gmb */
    vnd Ids, Gm, Gds, Gmb, cdrain;
    {
        vnd invVASCBE = one / VASCBE;
        vnd T9 = diffVds * invVASCBE;
        vnd T0 = one + T9;
        Ids = Idsa * T0;
        vnd gm0  = T0 * dIdsa_dVg - Idsa * (dVdseff_dVg + T9 * dVASCBE_dVg) * invVASCBE;
        vnd gds0 = T0 * dIdsa_dVd + Idsa * (one - dVdseff_dVd - T9 * dVASCBE_dVd) * invVASCBE;
        vnd gmb0 = T0 * dIdsa_dVb - Idsa * (dVdseff_dVb + T9 * dVASCBE_dVb) * invVASCBE;

        vnd tmp1 = gds0 + gm0 * dVgsteff_dVd;
        vnd tmp2 = gmb0 + gm0 * dVgsteff_dVb;
        Gm  = (Ids * dVdseff_dVg + Vdseff * gm0) * dVgsteff_dVg;
        Gds = Ids * (dVdseff_dVd + dVdseff_dVg * dVgsteff_dVd) + Vdseff * tmp1;
        Gmb = (Ids * (dVdseff_dVb + dVdseff_dVg * dVgsteff_dVb) + Vdseff * tmp2) * dVbseff_dVb;
        cdrain = Ids * Vdseff;
    }

    /* Source End Velocity Limit (vtl > 0) */
    if (b->f_vtl) {
        vnd T12 = b->invLeff / CoxeffWovL;
        vnd T11 = T12 / Vgsteff;
        vnd T10 = -T11 / Vgsteff;
        vnd vs = cdrain * T11;
        vnd dvs_dVg = Gm * T11 + cdrain * T10 * dVgsteff_dVg;
        vnd dvs_dVd = Gds * T11 + cdrain * T10 * dVgsteff_dVd;
        vnd dvs_dVb = Gmb * T11 + cdrain * T10 * dVgsteff_dVb;
        vnd T1 = vs * b->inv_vtl_tfactor;
        vnm m_act = b->m_vtl & vn_gt(T1, z0);
        if (vn_any(m_act)) {
            vnd s2 = T1 * T1;
            vnd T2 = one + s2 * s2 * s2;
            vnd vs_safe = vn_sel(m_act, vs, one);
            vnd T3 = (T2 - one) * vn_set1(6.0) / vs_safe;
            /* T2^(-1/6) = exp((-1/6) * log(T2)) */
            vnd Fsevl = vn_exp(vn_set1(-1.0 / 6.0) * vn_log_pos(T2));
            vnd T4 = vn_set1(-1.0 / 6.0) * Fsevl / T2;
            vnd dF_g = T4 * T3 * dvs_dVg;
            vnd dF_d = T4 * T3 * dvs_dVd;
            vnd dF_b = T4 * T3 * dvs_dVb;
            Gm     = vn_sel(m_act, Gm * Fsevl + cdrain * dF_g, Gm);
            Gmb    = vn_sel(m_act, Gmb * Fsevl + cdrain * dF_b, Gmb);
            Gds    = vn_sel(m_act, Gds * Fsevl + cdrain * dF_d, Gds);
            cdrain = vn_sel(m_act, cdrain * Fsevl, cdrain);
        }
    }

    /* ===================================================================== */
    /* 8. GIDL & GISL                                                        */
    /* ===================================================================== */
    vnd Igidl = z0, Ggidld = z0, Ggidlg = z0, Ggidlb = z0;
    if (b->f_gidl_en) {
        vnd T1 = (vds - vgs_eff - b->egidl) * b->inv_3toxe;
        vnm m_act = b->m_gidl_en & vn_gt(T1, z0) & vn_le(vbd, z0);
        if (vn_any(m_act)) {
            vnd T1s = vn_max(T1, vn_set1(1.0e-30));
            vnd dT1_dVd = b->inv_3toxe;
            vnd dT1_dVg = -dvgs_eff_dvg * dT1_dVd;
            vnd T2 = b->bgidl / T1s;
            vnm m_e = m_act & vn_lt(T2, vn_set1(100.0));
            vnd ig_lo = b->agidl_weffCJ * vn_set1(3.720075976e-44);
            vnd gd_l = ig_lo * dT1_dVd;
            vnd gg_l = ig_lo * dT1_dVg;
            vnd ig = ig_lo * T1s;
            vnd gd = gd_l, gg = gg_l;
            if (vn_any(m_e)) {
                vnd ige = b->agidl_weffCJ * T1s * vn_exp_clamped(-T2, -100.0, 0.0);
                vnd T3 = ige * (one + T2) / T1s;
                ig = vn_sel(m_e, ige, ig);
                gd = vn_sel(m_e, T3 * dT1_dVd, gd);
                gg = vn_sel(m_e, T3 * dT1_dVg, gg);
            }
            vnd T4 = vbd * vbd;
            vnd T5 = -vbd * T4;
            vnd T6 = b->cgidl + T5;
            vnd invT6 = one / T6;
            vnd T7 = T5 * invT6;
            vnd T8 = vn_set1(3.0) * b->cgidl * T4 * invT6 * invT6;
            Ggidld = vn_sel(m_act, gd * T7 + ig * T8, z0);
            Ggidlg = vn_sel(m_act, gg * T7, z0);
            Ggidlb = vn_sel(m_act, -ig * T8, z0);
            Igidl  = vn_sel(m_act, ig * T7, z0);
        }
    }

    vnd Igisl = z0, Ggisls = z0, Ggislg = z0, Ggislb = z0;
    if (b->f_gisl_en) {
        vnd T1 = (-vds - vgd_eff - b->egisl) * b->inv_3toxe;
        vnm m_act = b->m_gisl_en & vn_gt(T1, z0) & vn_le(vbs, z0);
        if (vn_any(m_act)) {
            vnd T1s = vn_max(T1, vn_set1(1.0e-30));
            vnd dT1_dVd = b->inv_3toxe;
            vnd dT1_dVg = -dvgd_eff_dvg * dT1_dVd;
            vnd T2 = b->bgisl / T1s;
            vnm m_e = m_act & vn_lt(T2, vn_set1(100.0));
            vnd ig_lo = b->agisl_weffCJ * vn_set1(3.720075976e-44);
            vnd gs_l = ig_lo * dT1_dVd;
            vnd gg_l = ig_lo * dT1_dVg;
            vnd ig = ig_lo * T1s;
            vnd gs = gs_l, gg = gg_l;
            if (vn_any(m_e)) {
                vnd ige = b->agisl_weffCJ * T1s * vn_exp_clamped(-T2, -100.0, 0.0);
                vnd T3 = ige * (one + T2) / T1s;
                ig = vn_sel(m_e, ige, ig);
                gs = vn_sel(m_e, T3 * dT1_dVd, gs);
                gg = vn_sel(m_e, T3 * dT1_dVg, gg);
            }
            vnd T4 = vbs * vbs;
            vnd T5 = -vbs * T4;
            vnd T6 = b->cgisl + T5;
            vnd invT6 = one / T6;
            vnd T7 = T5 * invT6;
            vnd T8 = vn_set1(3.0) * b->cgisl * T4 * invT6 * invT6;
            Ggisls = vn_sel(m_act, gs * T7 + ig * T8, z0);
            Ggislg = vn_sel(m_act, gg * T7, z0);
            Ggislb = vn_sel(m_act, -ig * T8, z0);
            Igisl  = vn_sel(m_act, ig * T7, z0);
        }
    }

    /* Scale DC currents/conductances by nf */
    if (b->f_nf_ne1) {
        cdrain = cdrain * b->nf;
        Gds = Gds * b->nf;
        Gm  = Gm  * b->nf;
        Gmb = Gmb * b->nf;
        Gbb = Gbb * b->nf;
        Gbg = Gbg * b->nf;
        Gbd = Gbd * b->nf;
        Isub = Isub * b->nf;
        Igidl = Igidl * b->nf; Ggidld = Ggidld * b->nf; Ggidlg = Ggidlg * b->nf; Ggidlb = Ggidlb * b->nf;
        Igisl = Igisl * b->nf; Ggisls = Ggisls * b->nf; Ggislg = Ggislg * b->nf; Ggislb = Ggislb * b->nf;
    }

    /* ===================================================================== */
    /* 1. Source/Drain Junction Diodes (dioMod == 1)                         */
    /* ===================================================================== */
    vnd gbs, cbs, gbd, cbd;
    {
        vnd T2 = vbs * b->invNvtms;
        vnm m_low = vn_lt(T2, nexp_th);
        vnm m_mid = (~m_low) & vn_le(vbs, b->vjsmFwd) & b->m_src_sat_pos;
        vnd evbs = z0;
        if (vn_any(m_mid))
            evbs = vn_exp_clamped(T2, -B4V_EXP_THRESHOLD, 100.0);
        vnd g_low = gmin;
        vnd c_low = b->SourceSatCurrent * (min_exp - one) + gmin * vbs;
        vnd g_mid = b->SourceSatCurrent * evbs * b->invNvtms + gmin;
        vnd c_mid = b->SourceSatCurrent * (evbs - one) + gmin * vbs;
        vnd T0h = b->IVjsmFwd * b->invNvtms;
        vnd g_hi = T0h + gmin;
        vnd c_hi = b->IVjsmFwd - b->SourceSatCurrent + T0h * (vbs - b->vjsmFwd) + gmin * vbs;
        gbs = vn_sel(m_low, g_low, vn_sel(m_mid, g_mid, g_hi));
        cbs = vn_sel(m_low, c_low, vn_sel(m_mid, c_mid, c_hi));
        gbs = vn_sel(b->m_src_sat_pos, gbs, gmin);
        cbs = vn_sel(b->m_src_sat_pos, cbs, gmin * vbs);
    }
    {
        vnd T2 = vbd * b->invNvtmd;
        vnm m_low = vn_lt(T2, nexp_th);
        vnm m_mid = (~m_low) & vn_le(vbd, b->vjdmFwd) & b->m_drn_sat_pos;
        vnd evbd = z0;
        if (vn_any(m_mid))
            evbd = vn_exp_clamped(T2, -B4V_EXP_THRESHOLD, 100.0);
        vnd g_low = gmin;
        vnd c_low = b->DrainSatCurrent * (min_exp - one) + gmin * vbd;
        vnd g_mid = b->DrainSatCurrent * evbd * b->invNvtmd + gmin;
        vnd c_mid = b->DrainSatCurrent * (evbd - one) + gmin * vbd;
        vnd T0h = b->IVjdmFwd * b->invNvtmd;
        vnd g_hi = T0h + gmin;
        vnd c_hi = b->IVjdmFwd - b->DrainSatCurrent + T0h * (vbd - b->vjdmFwd) + gmin * vbd;
        gbd = vn_sel(m_low, g_low, vn_sel(m_mid, g_mid, g_hi));
        cbd = vn_sel(m_low, c_low, vn_sel(m_mid, c_mid, c_hi));
        gbd = vn_sel(b->m_drn_sat_pos, gbd, gmin);
        cbd = vn_sel(b->m_drn_sat_pos, cbd, gmin * vbd);
    }
    if (b->has_trap_tunnel) {
        if (b->f_tt_s)
            vn_trap_tunnel_term(b->m_tt_s, vbs, b->vtss, b->invNvtmrs_s, b->SjctRevSat, &gbs, &cbs);
        if (b->f_tt_d)
            vn_trap_tunnel_term(b->m_tt_d, vbd, b->vtsd, b->invNvtmrs_d, b->DjctRevSat, &gbd, &cbd);
        if (b->f_tt_ssw)
            vn_trap_tunnel_term(b->m_tt_ssw, vbs, b->vtssws, b->invNvtmrssw_s, b->SswRevSat, &gbs, &cbs);
        if (b->f_tt_dsw)
            vn_trap_tunnel_term(b->m_tt_dsw, vbd, b->vtsswd, b->invNvtmrssw_d, b->DswRevSat, &gbd, &cbd);
        if (b->f_tt_sswg)
            vn_trap_tunnel_term(b->m_tt_sswg, vbs, b->vtsswgs, b->invNvtmrsswg_s, b->SswgRevSat, &gbs, &cbs);
        if (b->f_tt_dswg)
            vn_trap_tunnel_term(b->m_tt_dswg, vbd, b->vtsswgd, b->invNvtmrsswg_d, b->DswgRevSat, &gbd, &cbd);
    }

    /* ===================================================================== */
    {
    vnd ceqjs = (cbs - gbs * vbs) * b->type;
    vnd ceqjd = (cbd - gbd * vbd) * b->type;

    /* Forward vs Reverse DC stamping terms */
    vnd Gm_st   = vn_sel(m_fwd, Gm, -Gm);
    vnd Gmbs_st = vn_sel(m_fwd, Gmb, -Gmb);
    vnd FwdSum  = vn_sel(m_fwd, Gm + Gmb, z0);
    vnd RevSum  = vn_sel(m_fwd, z0, -(Gm_st + Gmbs_st)); /* == Gm + Gmb when rev */

    vnd ceqdrn_f = b->type * (cdrain - Gds * vds - Gm * vgs - Gmb * vbs);
    vnd ceqdrn_r = -b->type * (cdrain + Gds * vds - Gm * vgd - Gmb * vbd);
    vnd ceqdrn = vn_sel(m_fwd, ceqdrn_f, ceqdrn_r);

    vnd ceqbd_f = b->type * (Isub + Igidl - (Gbd + Ggidld) * vds - (Gbg + Ggidlg) * vgs - (Gbb + Ggidlb) * vbs);
    vnd ceqbs_f = b->type * (Igisl + Ggisls * vds - Ggislg * vgd - Ggislb * vbd);
    vnd ceqbd_r = b->type * (Igidl - Ggidld * vds - Ggidlg * vgs - Ggidlb * vbs);
    vnd ceqbs_r = b->type * (Isub + Igisl + (Gbd + Ggisls) * vds - (Gbg + Ggislg) * vgd - (Gbb + Ggislb) * vbd);
    vnd ceqbd = vn_sel(m_fwd, ceqbd_f, ceqbd_r);
    vnd ceqbs = vn_sel(m_fwd, ceqbs_f, ceqbs_r);

    vnd gb_sum = Gbd + Gbg + Gbb;
    vnd gbbdp  = vn_sel(m_fwd, -Gbd, gb_sum);
    vnd gbbsp  = vn_sel(m_fwd, gb_sum, -Gbd);
    vnd gbdpg  = vn_sel(m_fwd, Gbg, z0);
    vnd gbdpdp = vn_sel(m_fwd, Gbd, z0);
    vnd gbdpb  = vn_sel(m_fwd, Gbb, z0);
    vnd gbdpsp = vn_sel(m_fwd, -gb_sum, z0);
    vnd gbspg  = vn_sel(m_fwd, z0, Gbg);
    vnd gbspdp = vn_sel(m_fwd, z0, -gb_sum);
    vnd gbspb  = vn_sel(m_fwd, z0, Gbb);
    vnd gbspsp = vn_sel(m_fwd, z0, Gbd);

    vnd ggidl_sum = Ggidlg + Ggidld + Ggidlb;
    vnd ggisl_sum = Ggislg + Ggisls + Ggislb;

    vnd vm = b->m;
    l2->mdc44 = vm * (b->drainCond + Gds + gbd + RevSum + gbdpdp + Ggidld);
    l2->mdc46 = vm * (Gm_st + gbdpg + Ggidlg);
    l2->mdc47 = vm * (Gds + FwdSum - gbdpsp + ggidl_sum);
    l2->mdc48 = vm * (gbd - Gmbs_st - gbdpb - Ggidlb);
    l2->mdc51 = vm * (Gds + RevSum - gbspdp + ggisl_sum);
    l2->mdc52 = vm * (-Gm_st + gbspg + Ggislg);
    l2->mdc53 = vm * (b->sourceCond + Gds + gbs + FwdSum + gbspsp + Ggisls);
    l2->mdc55 = vm * (gbs + Gmbs_st - gbspb - Ggislb);
    l2->mdc58 = vm * (-gbd + gbbdp - Ggidld + ggisl_sum);
    l2->mdc59 = vm * (-Gbg - Ggidlg - Ggislg);
    l2->mdc60 = vm * (-gbs + gbbsp + ggidl_sum - Ggisls);
    l2->mdc61 = vm * (gbd + gbs - Gbb - Ggidlb - Ggislb);

    l2->mdc_rhsd = vm * (ceqjd - ceqbd - ceqdrn);
    l2->mdc_rhsb = vm * (ceqbd + ceqbs - ceqjd - ceqjs);
    l2->mdc_rhss = vm * (ceqdrn - ceqbs + ceqjs);

    /* Pre-assemble compact SoA convTest coefficients in dyn (exact match to b4v*cvtest.c) */
    {
        vnd Ggidls = -ggidl_sum;
        vnd Ggisld = -ggisl_sum;

        vnd id_f = cdrain + Isub - cbd + Igidl;
        vnd id_r = cdrain + cbd - Igidl;
        hot->Idtot = vn_sel(m_fwd, id_f, id_r);
        hot->cd_g  = vn_sel(m_fwd, Gm + Gbg + Ggidlg, Gm - Ggidlg);
        hot->cd_b  = vn_sel(m_fwd, Gmb + Gbb + Ggidlb - gbd, gbd + Gmb - Ggidlb);
        hot->cd_d  = vn_sel(m_fwd, Gds + Gbd + Ggidld + gbd, -(gbd + Gmb + Gm + Gds + Ggidls));

        hot->Ibtot = cbs + cbd - Igidl - Igisl - Isub;
        hot->cb_g  = -(Gbg + Ggidlg + Ggislg);
        hot->cb_b  = gbd + gbs - (Gbb + Ggidlb + Ggislb);
        vnd cbd_f  = -gbd - (Gbd + Ggidld) + Ggislg + Ggislb + Ggisls;
        vnd cbd_r  = -gbd + (Gbb + Ggislb) + (Gbg + Ggislg) + (Gbd + Ggisld - Ggidls);
        hot->cb_d  = vn_sel(m_fwd, cbd_f, cbd_r);
    }

    /* Only scatter 24 AoS instance fields when explicitly requested (e.g., .save @m[...]) */
    if (need_aos) {
        for (int lane = 0; lane < hot->n_active; lane++)
            out->mode[lane] = vn_mask_bit(m_fwd, lane) ? 1 : -1;
        vn_store(out->cd, cdrain);
        vn_store(out->gds, Gds);
        vn_store(out->gm, Gm);
        vn_store(out->gmbs, Gmb);
        vn_store(out->gbd, gbd);
        vn_store(out->gbs, gbs);
        vn_store(out->cbs, cbs);
        vn_store(out->cbd, cbd);
        vn_store(out->csub, Isub);
        vn_store(out->gbds, Gbd);
        vn_store(out->gbgs, Gbg);
        vn_store(out->gbbs, Gbb);
        vn_store(out->Igidl, Igidl);
        vn_store(out->ggidld, Ggidld);
        vn_store(out->ggidlg, Ggidlg);
        vn_store(out->ggidlb, Ggidlb);
        vn_store(out->Igisl, Igisl);
        vn_store(out->ggisls, Ggisls);
        vn_store(out->ggislg, Ggislg);
        vn_store(out->ggislb, Ggislb);
    }
    }

    /* === PHASE B: Intrinsic C-V + Junction Cap + Overlap Cap (all DC/Diode regs dead!) === */

    /* 9. Intrinsic C-V (capMod == 2, Charge-Thickness Model)                */
    /* ===================================================================== */
    /* Note: if Vbseff < 0 -> VbseffCV = Vbseff, dVbseffCV_dVb = 1.0;
     *       else -> VbseffCV = phi - Phis = Vbseff, dVbseffCV_dVb = -(-1.0) = 1.0! */
    vnd VbseffCV = Vbseff;
    vnd VgsteffCV, dVgCV_dVg, dVgCV_dVd, dVgCV_dVb;
    {
        vnd noff = n * b->noff;
        vnd dnoff_dVd = b->noff * dn_dVd;
        vnd dnoff_dVb = b->noff * dn_dVb;
        vnd T0 = b->vtm * noff;
        vnd dVg_off = Vgst - b->voffcv;
        vnd VgstNVt = dVg_off / T0;
        vnm m_hi = vn_gt(VgstNVt, exp_th);
        vnm m_lo = vn_lt(VgstNVt, nexp_th);
        vnm m_mid = (~m_hi) & (~m_lo);

        vnd vg_lo = T0 * vn_set1(B4V_LOG1P_MIN_EXP);
        vnd inv_noff = one / noff;
        vnd ratio_lo = vg_lo * inv_noff;
        VgsteffCV = vn_sel(m_hi, dVg_off, vg_lo);
        dVgCV_dVg = vn_sel(m_hi, dVgs_eff_dVg, z0);
        dVgCV_dVd = vn_sel(m_hi, -dVth_dVd, ratio_lo * dnoff_dVd);
        dVgCV_dVb = vn_sel(m_hi, -dVth_dVb, ratio_lo * dnoff_dVb);
        if (vn_any(m_mid)) {
            vnd ex = vn_exp_clamped(VgstNVt, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            vnd vg = T0 * vn_log_pos(one + ex);
            vnd dg0 = ex / (one + ex);
            VgsteffCV = vn_sel(m_mid, vg, VgsteffCV);
            dVgCV_dVd = vn_sel(m_mid, -dg0 * (dVth_dVd + dVg_off * inv_noff * dnoff_dVd) + vg * inv_noff * dnoff_dVd, dVgCV_dVd);
            dVgCV_dVb = vn_sel(m_mid, -dg0 * (dVth_dVb + dVg_off * inv_noff * dnoff_dVb) + vg * inv_noff * dnoff_dVb, dVgCV_dVb);
            dVgCV_dVg = vn_sel(m_mid, dg0 * dVgs_eff_dVg, dVgCV_dVg);
        }
    }

    vnd qgate, qbulk, qdrn;
    vnd cggb, cgsb, cgdb, cdgb, cdsb, cddb, cbgb, cbsb, cbdb;
    {
        vnd V3 = b->vfbzb - Vgs_eff + VbseffCV - vn_set1(B4V_DELTA_3);
        vnd f4 = vn_sel(b->m_vfbzb_le0, vn_set1(-4.0 * B4V_DELTA_3), vn_set1(4.0 * B4V_DELTA_3));
        vnd T0_fb = vn_sqrt(V3 * V3 + f4 * b->vfbzb);
        vnd T1_fb = half * (one + V3 / T0_fb);
        vnd Vfbeff = b->vfbzb - half * (V3 + T0_fb);
        vnd dVfbeff_dVg = T1_fb * dVgs_eff_dVg;
        vnd dVfbeff_dVb = -T1_fb;

        vnd T0_ox = (Vgs_eff - VbseffCV - b->vfbzb) * b->inv_tox_ctm;
        vnd dT0_ox_dVg = dVgs_eff_dVg * b->inv_tox_ctm;
        vnd dT0_ox_dVb = -b->inv_tox_ctm;
        vnd tmp_ac = T0_ox * b->acde;
        vnm m_ac_lo = vn_le(tmp_ac, nexp_th);
        vnm m_ac_hi = vn_ge(tmp_ac, exp_th);
        vnm m_ac_mid = (~m_ac_lo) & (~m_ac_hi);
        vnd Tcen = b->ldeb * vn_sel(m_ac_lo, min_exp, max_exp);
        vnd dTcen_dVg = z0, dTcen_dVb = z0;
        if (vn_any(m_ac_mid)) {
            vnd tc = b->ldeb * vn_exp_clamped(tmp_ac, -B4V_EXP_THRESHOLD, B4V_EXP_THRESHOLD);
            vnd dtc = b->acde * tc;
            Tcen = vn_sel(m_ac_mid, tc, Tcen);
            dTcen_dVb = vn_sel(m_ac_mid, dtc * dT0_ox_dVb, z0);
            dTcen_dVg = vn_sel(m_ac_mid, dtc * dT0_ox_dVg, z0);
        }

        vnd V3l = b->ldeb - Tcen - b->link_ctm;
        vnd V4l = vn_sqrt(V3l * V3l + vn_set1(4.0) * b->link_ctm * b->ldeb);
        Tcen = b->ldeb - half * (V3l + V4l);
        vnd T1l = half * (one + V3l / V4l);
        dTcen_dVg = dTcen_dVg * T1l;
        dTcen_dVb = dTcen_dVb * T1l;

        vnd Ccen = epssi / Tcen;
        vnd T2c = b->coxp / (b->coxp + Ccen);
        vnd Coxeff = T2c * Ccen;
        vnd dCox_base = T2c * T2c * (-Ccen / Tcen);
        vnd dCoxeff_dVg = dCox_base * dTcen_dVg;
        vnd dCoxeff_dVb = dCox_base * dTcen_dVb;
        vnd CoxWLcen = b->CoxWL_ov_coxe * Coxeff;

        vnd Qac0 = CoxWLcen * (Vfbeff - b->vfbzb);
        vnd QovCox_ac = Qac0 / Coxeff;
        vnd dQac0_dVg = CoxWLcen * dVfbeff_dVg + QovCox_ac * dCoxeff_dVg;
        vnd dQac0_dVb = CoxWLcen * dVfbeff_dVb + QovCox_ac * dCoxeff_dVb;

        vnd T0s = half * b->k1ox;
        vnd T3s = Vgs_eff - Vfbeff - VbseffCV - VgsteffCV;
        vnm m_t3neg = vn_lt(T3s, z0);
        vnd T1_pos = vn_sqrt(vn_max(T0s * T0s + T3s, vn_set1(1.0e-300)));
        vnd T1_neg = T0s + T3s / vn_sel(b->m_k1ox_zero, one, b->k1ox);
        vnd T1s = vn_sel(b->m_k1ox_zero, z0, vn_sel(m_t3neg, T1_neg, T1_pos));
        vnd T2s = vn_sel(b->m_k1ox_zero, z0, vn_sel(m_t3neg, CoxWLcen, CoxWLcen * T0s / T1_pos));
        vnd Qsub0 = CoxWLcen * b->k1ox * (T1s - T0s);
        vnd QovCox_sub = Qsub0 / Coxeff;
        vnd dQsub0_dVg = T2s * (dVgs_eff_dVg - dVfbeff_dVg - dVgCV_dVg) + QovCox_sub * dCoxeff_dVg;
        vnd dQsub0_dVd = -T2s * dVgCV_dVd;
        vnd dQsub0_dVb = -T2s * (dVfbeff_dVb + one + dVgCV_dVb) + QovCox_sub * dCoxeff_dVb;

        /* DeltaPhi & VgDP */
        vnd Den_dp = vn_sel(b->m_k1ox_le0, vn_set1(0.25) * b->moin * b->vtm,
                            b->moin * b->vtm * b->k1ox * b->k1ox);
        vnd T0_dp = vn_sel(b->m_k1ox_le0, half * b->sqrtPhi, b->k1ox * b->sqrtPhi);
        vnd T1_dp = two * T0_dp + VgsteffCV;
        vnd DeltaPhi = b->vtm * vn_log_pos(one + T1_dp * VgsteffCV / Den_dp);
        vnd dDeltaPhi_dVg = two * b->vtm * (T1_dp - T0_dp) / (Den_dp + T1_dp * VgsteffCV);

        vnd T0_vg = VgsteffCV - DeltaPhi - vn_set1(0.001);
        vnd dT0_vg = one - dDeltaPhi_dVg;
        vnd T1_vg = vn_sqrt(T0_vg * T0_vg + VgsteffCV * vn_set1(0.004));
        vnd VgDP = half * (T0_vg + T1_vg);
        vnd dVgDP_dVg = half * (dT0_vg + (T0_vg * dT0_vg + vn_set1(0.002)) / T1_vg);

        /* Inversion Tcen & Coxeff */
        vnd T0_i = (VgsteffCV + b->vtfbphi2) * b->inv_2e8_toxp;
        vnd tmp_i = vn_exp(b->bdos0p7 * vn_log_pos(T0_i));
        vnd T1_i = one + tmp_i;
        vnd invT1_i = one / T1_i;
        vnd T2_i = b->bdos0p7 * tmp_i * b->inv_2e8_toxp / T0_i;
        vnd Tcen_i = b->ados1p9 * invT1_i;
        vnd dTcen_i0 = -Tcen_i * T2_i * invT1_i;
        vnd dTcen_id = dTcen_i0 * dVgCV_dVd;
        vnd dTcen_ib = dTcen_i0 * dVgCV_dVb;
        vnd dTcen_ig = dTcen_i0 * dVgCV_dVg;

        vnd Ccen_i = epssi / Tcen_i;
        vnd T0_ci = b->coxp / (b->coxp + Ccen_i);
        vnd Coxeff_i = T0_ci * Ccen_i;
        vnd dCox_i0 = T0_ci * T0_ci * (-Ccen_i / Tcen_i);
        vnd dCoxeff_id = dCox_i0 * dTcen_id;
        vnd dCoxeff_ib = dCox_i0 * dTcen_ib;
        vnd dCoxeff_ig = dCox_i0 * dTcen_ig;
        vnd CoxWLcen_i = b->CoxWL_ov_coxe * Coxeff_i;

        /* VdsatCV & VdseffCV */
        vnd AbulkCV = Abulk0 * b->abulkCVfactor;
        vnd dAbulkCV_dVb = b->abulkCVfactor * dAbulk0_dVb;
        vnd invAbCV = one / AbulkCV;
        vnd VdsatCV = VgDP * invAbCV;

        vnd T0_v = VdsatCV - Vds - vn_set1(B4V_DELTA_4);
        vnd dT0_vg_cv = dVgDP_dVg * invAbCV;
        vnd dT0_vb_cv = -VdsatCV * dAbulkCV_dVb * invAbCV;
        vnd T1_v = vn_sqrt(T0_v * T0_v + vn_set1(4.0 * B4V_DELTA_4) * VdsatCV);
        vnd invT1_v = one / T1_v;
        vnd dT1_base = (T0_v + vn_set1(2.0 * B4V_DELTA_4)) * invT1_v;
        vnd dT1_vd_cv = -T0_v * invT1_v;
        vnd dT1_vb_cv = dT1_base * dT0_vb_cv;
        vnd dT1_vg_cv = dT1_base * dT0_vg_cv;

        vnm m_ge = vn_ge(T0_v, z0);
        vnd inv_diff_v = one / (T1_v - T0_v);
        vnd T3_v = vn_set1(2.0 * B4V_DELTA_4) * inv_diff_v;
        vnd T4_v = one - T3_v;
        vnd T5_v = VdsatCV * T3_v * inv_diff_v;

        vnd VdseffCV     = vn_sel(m_ge, VdsatCV - half * (T0_v + T1_v), VdsatCV * T4_v);
        vnd dVdseffCV_dVg = vn_sel(m_ge, half * (dT0_vg_cv - dT1_vg_cv), dT0_vg_cv * T4_v + T5_v * (dT1_vg_cv - dT0_vg_cv));
        vnd dVdseffCV_dVd = vn_sel(m_ge, half * (one - dT1_vd_cv), T5_v * (dT1_vd_cv + one));
        vnd vb_fac = b->is_v6 ? (T4_v - T5_v) : (one - T5_v);
        vnd dVdseffCV_dVb = vn_sel(m_ge, half * (dT0_vb_cv - dT1_vb_cv), dT0_vb_cv * vb_fac + T5_v * dT1_vb_cv);

        vnm m_vds0 = vn_eq(Vds, z0);
        VdseffCV      = vn_sel(m_vds0, z0, VdseffCV);
        dVdseffCV_dVg = vn_sel(m_vds0, z0, dVdseffCV_dVg);
        dVdseffCV_dVb = vn_sel(m_vds0, z0, dVdseffCV_dVb);

        /* Channel charges qgate, qbulk, qsrc */
        vnd T0_q = AbulkCV * VdseffCV;
        vnd T1_q = VgDP;
        vnd T2_q = vn_set1(12.0) * (T1_q - half * T0_q + vn_set1(1.0e-20));
        vnd invT2_q = one / T2_q;
        vnd T3_q = T0_q * invT2_q;
        vnd T4_q = one - vn_set1(12.0) * T3_q * T3_q;
        vnd T5_q = AbulkCV * (vn_set1(6.0) * T0_q * (vn_set1(4.0) * T1_q - T0_q) * (invT2_q * invT2_q) - half);
        vnd T6_q = T5_q * VdseffCV * invAbCV;

        vnd invCox_i = one / Coxeff_i;
        qgate = CoxWLcen_i * (T1_q - T0_q * (half - T3_q));
        vnd QovCox_g = qgate * invCox_i;
        vnd Cgg1_0 = CoxWLcen_i * (T4_q * dVgDP_dVg + T5_q * dVdseffCV_dVg);
        vnd Cgd1 = CoxWLcen_i * T5_q * dVdseffCV_dVd + Cgg1_0 * dVgCV_dVd + QovCox_g * dCoxeff_id;
        vnd Cgb1 = CoxWLcen_i * (T5_q * dVdseffCV_dVb + T6_q * dAbulkCV_dVb) + Cgg1_0 * dVgCV_dVb + QovCox_g * dCoxeff_ib;
        vnd Cgg1 = Cgg1_0 * dVgCV_dVg + QovCox_g * dCoxeff_ig;

        vnd T7_q = one - AbulkCV;
        vnd T8_q = T2_q * T2_q;
        vnd T9_q = vn_set1(12.0) * T7_q * T0_q * T0_q / (T8_q * AbulkCV);
        vnd T10_q = T9_q * dVgDP_dVg;
        vnd T11_q = -T7_q * T5_q * invAbCV;
        vnd T12_q = -(T9_q * T1_q * invAbCV + VdseffCV * (half - T3_q));

        qbulk = CoxWLcen_i * T7_q * (half * VdseffCV - T0_q * VdseffCV * invT2_q);
        vnd QovCox_b = qbulk * invCox_i;
        vnd Cbg1_0 = CoxWLcen_i * (T10_q + T11_q * dVdseffCV_dVg);
        vnd Cbd1 = CoxWLcen_i * T11_q * dVdseffCV_dVd + Cbg1_0 * dVgCV_dVd + QovCox_b * dCoxeff_id;
        vnd Cbb1 = CoxWLcen_i * (T11_q * dVdseffCV_dVb + T12_q * dAbulkCV_dVb) + Cbg1_0 * dVgCV_dVb + QovCox_b * dCoxeff_ib;
        vnd Cbg1 = Cbg1_0 * dVgCV_dVg + QovCox_b * dCoxeff_ig;

        vnd qsrc, Csg, Csd, Csb;
        if (b->xpart_mode == 0) {
            /* All 8 lanes use 40/60 partition (standard GF180 xpart=0) */
            vnd T2p = T2_q * vn_set1(1.0 / 12.0);
            vnd invT2p = one / T2p;
            vnd T3p = half * CoxWLcen_i * (invT2p * invT2p);
            vnd T4p = T1_q * (vn_set1(2.0 / 3.0) * T0_q * T0_q + T1_q * (T1_q - vn_set1(4.0 / 3.0) * T0_q))
                    - vn_set1(2.0 / 15.0) * T0_q * T0_q * T0_q;
            qsrc = -T3p * T4p;
            vnd QovCox_s = qsrc * invCox_i;
            vnd T8p = vn_set1(4.0 / 3.0) * T1_q * (T1_q - T0_q) + vn_set1(0.4) * T0_q * T0_q;
            vnd T5p = -two * qsrc * invT2p - T3p * (T1_q * (vn_set1(3.0) * T1_q - vn_set1(8.0 / 3.0) * T0_q)
                    + vn_set1(2.0 / 3.0) * T0_q * T0_q);
            vnd T6p = AbulkCV * (qsrc * invT2p + T3p * T8p);
            vnd T7p = T6p * VdseffCV * invAbCV;
            vnd Csg0 = T5p * dVgDP_dVg + T6p * dVdseffCV_dVg;
            Csd = Csg0 * dVgCV_dVd + T6p * dVdseffCV_dVd + QovCox_s * dCoxeff_id;
            Csb = Csg0 * dVgCV_dVb + T6p * dVdseffCV_dVb + T7p * dAbulkCV_dVb + QovCox_s * dCoxeff_ib;
            Csg = Csg0 * dVgCV_dVg + QovCox_s * dCoxeff_ig;
        } else if (b->xpart_mode == 2) {
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
            if (b->f_xpart_gt05) {
                vnd qs0 = -CoxWLcen_i * (half * T1_q + vn_set1(0.25) * T0_q - half * T0_q * T0_q * invT2_q);
                vnd Qov = qs0 * invCox_i;
                vnd T2d = T2_q + T2_q;
                vnd invT3d = one / (T2d * T2d);
                vnd T7d = -(vn_set1(0.25) - vn_set1(12.0) * T0_q * (vn_set1(4.0) * T1_q - T0_q) * invT3d);
                vnd T4d = -(half + vn_set1(24.0) * T0_q * T0_q * invT3d) * dVgDP_dVg;
                vnd T5d = T7d * AbulkCV;
                vnd T6d = T7d * VdseffCV;
                vnd Csg0 = CoxWLcen_i * (T4d + T5d * dVdseffCV_dVg);
                qsrc = vn_sel(b->m_xpart_gt05, qs0, qsrc);
                Csd  = vn_sel(b->m_xpart_gt05, CoxWLcen_i * T5d * dVdseffCV_dVd + Csg0 * dVgCV_dVd + Qov * dCoxeff_id, Csd);
                Csb  = vn_sel(b->m_xpart_gt05, CoxWLcen_i * (T5d * dVdseffCV_dVb + T6d * dAbulkCV_dVb) + Csg0 * dVgCV_dVb + Qov * dCoxeff_ib, Csb);
                Csg  = vn_sel(b->m_xpart_gt05, Csg0 * dVgCV_dVg + Qov * dCoxeff_ig, Csg);
            }
            if (b->f_xpart_lt05) {
                vnd T2p = T2_q * vn_set1(1.0 / 12.0);
                vnd invT2p = one / T2p;
                vnd T3p = half * CoxWLcen_i * (invT2p * invT2p);
                vnd T4p = T1_q * (vn_set1(2.0 / 3.0) * T0_q * T0_q + T1_q * (T1_q - vn_set1(4.0 / 3.0) * T0_q))
                        - vn_set1(2.0 / 15.0) * T0_q * T0_q * T0_q;
                vnd qs4 = -T3p * T4p;
                vnd Qov = qs4 * invCox_i;
                vnd T8p = vn_set1(4.0 / 3.0) * T1_q * (T1_q - T0_q) + vn_set1(0.4) * T0_q * T0_q;
                vnd T5p = -two * qs4 * invT2p - T3p * (T1_q * (vn_set1(3.0) * T1_q - vn_set1(8.0 / 3.0) * T0_q)
                        + vn_set1(2.0 / 3.0) * T0_q * T0_q);
                vnd T6p = AbulkCV * (qs4 * invT2p + T3p * T8p);
                vnd T7p = T6p * VdseffCV * invAbCV;
                vnd Csg0 = T5p * dVgDP_dVg + T6p * dVdseffCV_dVg;
                qsrc = vn_sel(b->m_xpart_lt05, qs4, qsrc);
                Csd  = vn_sel(b->m_xpart_lt05, Csg0 * dVgCV_dVd + T6p * dVdseffCV_dVd + Qov * dCoxeff_id, Csd);
                Csb  = vn_sel(b->m_xpart_lt05, Csg0 * dVgCV_dVb + T6p * dVdseffCV_dVb + T7p * dAbulkCV_dVb + Qov * dCoxeff_ib, Csb);
                Csg  = vn_sel(b->m_xpart_lt05, Csg0 * dVgCV_dVg + Qov * dCoxeff_ig, Csg);
            }
        }

        qgate = qgate + Qac0 + Qsub0 - qbulk;
        qbulk = qbulk - (Qac0 + Qsub0);
        qdrn  = -(qgate + qbulk + qsrc);

        vnd Cbg = Cbg1 - dQac0_dVg - dQsub0_dVg;
        vnd Cbd = Cbd1 - dQsub0_dVd;
        vnd Cbb = (Cbb1 - dQac0_dVb - dQsub0_dVb) * dVbseff_dVb;
        vnd Cgg = Cgg1 - Cbg;
        vnd Cgd = Cgd1 - Cbd;
        vnd Cgb = (Cgb1 - (Cbb1 - dQac0_dVb - dQsub0_dVb)) * dVbseff_dVb;
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
    vnd qbs, capbs, qbd, capbd;
    {
        /* Source-bulk */
        vnm m_neg = vn_lt(vbs, z0);
        vnd T1_f = vbs * b->jctS_lin;
        qbs   = vbs * (b->jctS_sum + half * T1_f);
        capbs = b->jctS_sum + T1_f;
        if (vn_any(m_neg)) {
            vnd q_n = z0, c_n = z0;
            if (b->f_czbs && vn_any(b->m_czbs & m_neg)) {
                vnm m = b->m_czbs & m_neg;
                vnd arg = vn_sel(m, one - vbs * b->invPhiBS, one);
                vnd sarg;
                if (!vn_any((~b->m_mjs_half) & m))
                    sarg = one / vn_sqrt(arg);
                else {
                    sarg = vn_exp(-b->MJS * vn_log(arg));
                    if (b->f_mjs_half && vn_any(b->m_mjs_half & m))
                        sarg = vn_sel(b->m_mjs_half & m, one / vn_sqrt(arg), sarg);
                }
                q_n = vn_sel(m, q_n + b->PhiBS * b->czbs * (one - arg * sarg) * b->inv1mMJS, q_n);
                c_n = vn_sel(m, c_n + b->czbs * sarg, c_n);
            }
            if (b->f_czbssw && vn_any(b->m_czbssw & m_neg)) {
                vnm m = b->m_czbssw & m_neg;
                vnd arg = vn_sel(m, one - vbs * b->invPhiBSWS, one);
                vnd sarg;
                if (!vn_any((~b->m_mjsws_half) & m))
                    sarg = one / vn_sqrt(arg);
                else {
                    sarg = vn_exp(-b->MJSWS * vn_log(arg));
                    if (b->f_mjsws_half && vn_any(b->m_mjsws_half & m))
                        sarg = vn_sel(b->m_mjsws_half & m, one / vn_sqrt(arg), sarg);
                }
                q_n = vn_sel(m, q_n + b->PhiBSWS * b->czbssw * (one - arg * sarg) * b->inv1mMJSWS, q_n);
                c_n = vn_sel(m, c_n + b->czbssw * sarg, c_n);
            }
            if (b->f_czbsswg && vn_any(b->m_czbsswg & m_neg)) {
                vnm m = b->m_czbsswg & m_neg;
                vnd arg = vn_sel(m, one - vbs * b->invPhiBSWGS, one);
                vnd sarg;
                if (!vn_any((~b->m_mjswgs_half) & m))
                    sarg = one / vn_sqrt(arg);
                else {
                    sarg = vn_exp(-b->MJSWGS * vn_log(arg));
                    if (b->f_mjswgs_half && vn_any(b->m_mjswgs_half & m))
                        sarg = vn_sel(b->m_mjswgs_half & m, one / vn_sqrt(arg), sarg);
                }
                q_n = vn_sel(m, q_n + b->PhiBSWGS * b->czbsswg * (one - arg * sarg) * b->inv1mMJSWGS, q_n);
                c_n = vn_sel(m, c_n + b->czbsswg * sarg, c_n);
            }
            qbs   = vn_sel(m_neg, q_n, qbs);
            capbs = vn_sel(m_neg, c_n, capbs);
        }
    }
    {
        /* Drain-bulk */
        vnm m_neg = vn_lt(vbd, z0);
        vnd T1_f = vbd * b->jctD_lin;
        qbd   = vbd * (b->jctD_sum + half * T1_f);
        capbd = b->jctD_sum + T1_f;
        if (vn_any(m_neg)) {
            vnd q_n = z0, c_n = z0;
            if (b->f_czbd && vn_any(b->m_czbd & m_neg)) {
                vnm m = b->m_czbd & m_neg;
                vnd arg = vn_sel(m, one - vbd * b->invPhiBD, one);
                vnd sarg;
                if (!vn_any((~b->m_mjd_half) & m))
                    sarg = one / vn_sqrt(arg);
                else {
                    sarg = vn_exp(-b->MJD * vn_log(arg));
                    if (b->f_mjd_half && vn_any(b->m_mjd_half & m))
                        sarg = vn_sel(b->m_mjd_half & m, one / vn_sqrt(arg), sarg);
                }
                q_n = vn_sel(m, q_n + b->PhiBD * b->czbd * (one - arg * sarg) * b->inv1mMJD, q_n);
                c_n = vn_sel(m, c_n + b->czbd * sarg, c_n);
            }
            if (b->f_czbdsw && vn_any(b->m_czbdsw & m_neg)) {
                vnm m = b->m_czbdsw & m_neg;
                vnd arg = vn_sel(m, one - vbd * b->invPhiBSWD, one);
                vnd sarg;
                if (!vn_any((~b->m_mjswd_half) & m))
                    sarg = one / vn_sqrt(arg);
                else {
                    sarg = vn_exp(-b->MJSWD * vn_log(arg));
                    if (b->f_mjswd_half && vn_any(b->m_mjswd_half & m))
                        sarg = vn_sel(b->m_mjswd_half & m, one / vn_sqrt(arg), sarg);
                }
                q_n = vn_sel(m, q_n + b->PhiBSWD * b->czbdsw * (one - arg * sarg) * b->inv1mMJSWD, q_n);
                c_n = vn_sel(m, c_n + b->czbdsw * sarg, c_n);
            }
            if (b->f_czbdswg && vn_any(b->m_czbdswg & m_neg)) {
                vnm m = b->m_czbdswg & m_neg;
                vnd arg = vn_sel(m, one - vbd * b->invPhiBSWGD, one);
                vnd sarg;
                if (!vn_any((~b->m_mjswgd_half) & m))
                    sarg = one / vn_sqrt(arg);
                else {
                    sarg = vn_exp(-b->MJSWGD * vn_log(arg));
                    if (b->f_mjswgd_half && vn_any(b->m_mjswgd_half & m))
                        sarg = vn_sel(b->m_mjswgd_half & m, one / vn_sqrt(arg), sarg);
                }
                q_n = vn_sel(m, q_n + b->PhiBSWGD * b->czbdswg * (one - arg * sarg) * b->inv1mMJSWGD, q_n);
                c_n = vn_sel(m, c_n + b->czbdswg * sarg, c_n);
            }
            qbd   = vn_sel(m_neg, q_n, qbd);
            capbd = vn_sel(m_neg, c_n, capbd);
        }
    }

    /* ===================================================================== */
    /* 11. Overlap Capacitances (capMod == 2) & Total Charge Assembly        */
    /* ===================================================================== */
    vnd cgdo, qgdo, cgso, qgso;
    {
        vnd T0 = vgd + vn_set1(B4V_DELTA_1);
        vnd T1 = vn_sqrt(T0 * T0 + vn_set1(4.0 * B4V_DELTA_1));
        vnd T2 = half * (T0 - T1);
        vnd T4 = vn_sqrt(one - vn_set1(4.0) * T2 * b->inv_ckappad);
        cgdo = b->cgdo + b->weffCV_cgdl - b->weffCV_cgdl * (one - one / T4) * (half - half * T0 / T1);
        qgdo = (b->cgdo + b->weffCV_cgdl) * vgd - b->weffCV_cgdl * (T2 + half * b->ckappad * (T4 - one));

        vnd T0s = vgs + vn_set1(B4V_DELTA_1);
        vnd T1s = vn_sqrt(T0s * T0s + vn_set1(4.0 * B4V_DELTA_1));
        vnd T2s = half * (T0s - T1s);
        vnd T4s = vn_sqrt(one - vn_set1(4.0) * T2s * b->inv_ckappas);
        cgso = b->cgso + b->weffCV_cgsl - b->weffCV_cgsl * (one - one / T4s) * (half - half * T0s / T1s);
        qgso = (b->cgso + b->weffCV_cgsl) * vgs - b->weffCV_cgsl * (T2s + half * b->ckappas * (T4s - one));

        if (b->f_nf_ne1) {
            cgdo = cgdo * b->nf;
            cgso = cgso * b->nf;
            qgdo = qgdo * b->nf;
            qgso = qgso * b->nf;
        }
    }

    vnd qgb = b->cgbo * vgb;
    vnd qgate_tot = qgate + qgdo + qgso + qgb;
    vnd qbulk_tot = qbulk - qgb;
    vnd qdrn_fwd  = qdrn - qgdo;
    vnd qsrc_fwd  = -(qgate_tot + qbulk_tot + qdrn_fwd);
    vnd qsrc_rev  = qdrn - qgso;
    vnd qdrn_rev  = -(qgate_tot + qbulk_tot + qsrc_rev);
    vnd qdrn_tot  = vn_sel(m_fwd, qdrn_fwd, qdrn_rev);
    vnd qsrc_tot  = vn_sel(m_fwd, qsrc_fwd, qsrc_rev);

    /* Unscaled (time-independent) terminal capacitances */
    vnd Cggb = cggb + cgdo + cgso + b->cgbo;
    vnd Cgdb = vn_sel(m_fwd, cgdb, cgsb) - cgdo;
    vnd Cgsb = vn_sel(m_fwd, cgsb, cgdb) - cgso;

    vnd cgg_bg_dg = cggb + cbgb + cdgb;
    vnd Cdgb = vn_sel(m_fwd, cdgb - cgdo, -(cgg_bg_dg + cgdo));
    vnd Csgb = vn_sel(m_fwd, -(cgg_bg_dg + cgso), cdgb - cgso);
    vnd Cbgb = cbgb - b->cgbo;

    vnd c_sum_d = cgdb + cbdb + cddb;
    vnd c_sum_s = cgsb + cbsb + cdsb;
    vnd Cddb = vn_sel(m_fwd, cddb + capbd + cgdo, capbd + cgdo - c_sum_s);
    vnd Cdsb = vn_sel(m_fwd, cdsb, -c_sum_d);
    vnd Csdb = vn_sel(m_fwd, -c_sum_d, cdsb);
    vnd Cssb = vn_sel(m_fwd, capbs + cgso - c_sum_s, cddb + capbs + cgso);

    vnd Cbdb = vn_sel(m_fwd, cbdb, cbsb) - capbd;
    vnd Cbsb = vn_sel(m_fwd, cbsb, cbdb) - capbs;

    /* ===================================================================== */
    /* 12. Cache Time-Independent Physics (DC + C + Charges) in dyn          */
    /* ===================================================================== */
    vnd qg_st = qgate_tot;
    vnd qd_st = qdrn_tot - qbd;
    vnd qs_st = qsrc_tot - qbs;
    vnd qb_st = qbulk_tot + qbd + qbs;

    vn_store(l2->a_vds, vds); vn_store(l2->a_vgs, vgs);
    vn_store(l2->a_vbs, vbs); vn_store(l2->a_vbd, vbd);
    vn_store(l2->a_qg, qg_st); vn_store(l2->a_qd, qd_st); vn_store(l2->a_qb, qb_st);
    l2->qb_st = qb_st; l2->qg_st = qg_st; l2->qd_st = qd_st;

    if (need_aos) {
        double a_qbs[B4N_W], a_qbd[B4N_W], a_qs[B4N_W];
        vn_store(a_qbs, qbs); vn_store(a_qbd, qbd); vn_store(a_qs, qs_st);
        for (int lane = 0; lane < hot->n_active; lane++) {
            int st = hot->stateBase[lane];
            s0[st + 4]  = l2->a_vbs[lane]; /* vdbs */
            s0[st + 5]  = l2->a_vbd[lane]; /* vdbd */
            s0[st + 6]  = l2->a_vbs[lane]; /* vsbs */
            s0[st + 7]  = l2->a_vgs[lane]; /* vges */
            s0[st + 8]  = l2->a_vgs[lane]; /* vgms */
            s0[st + 9]  = 0.0;              /* vses */
            s0[st + 10] = l2->a_vds[lane]; /* vdes */
            s0[st + 19] = a_qbs[lane];
            s0[st + 21] = a_qbd[lane];
            s0[st + 27] = 0.0;              /* qdef */
            s0[st + 28] = a_qs[lane];
        }
    }

    l2->Cggb = Cggb; l2->Cgdb = Cgdb; l2->Cgsb = Cgsb;
    l2->Cdgb = Cdgb; l2->Cddb = Cddb; l2->Cdsb = Cdsb;
    l2->Csgb = Csgb; l2->Csdb = Csdb; l2->Cssb = Cssb;
    l2->Cbgb = Cbgb; l2->Cbdb = Cbdb; l2->Cbsb = Cbsb;

    l2->q_lin_g = -Cggb * vgb + Cgdb * vbd + Cgsb * vbs;
    l2->q_lin_d = -Cdgb * vgb + Cddb * vbd + Cdsb * vbs;
    l2->q_lin_b = -Cbgb * vgb + Cbdb * vbd + Cbsb * vbs;

    /* ===================================================================== */
    /* 13. Integrate Charges & Assemble Matrix/RHS Stamps in l2->st_*       */
    /* ===================================================================== */
    bsim4_neon_integrate_and_assemble(hot, l2, vds, vgs, vbs, 0,
                                        ag0, ag1, ag2, is_gear2, is_trap2, s0, s1, s2);

}

#endif /* NGSPICE_OPT_APPLE */
#endif /* NGSPICE_BSIM4_NEON_CORE_H */
