/* Power capping API (implemented in powercap.c) */
#ifndef CUDAMPI_POWERCAP_H
#define CUDAMPI_POWERCAP_H

// Top-level configuration and application of power caps
void __cudampi__loadAndLogPowercapConfig(void);
void __cudampi__applyAllPowercaps(void);
void __cudampi__powercappingManagerStep(void);

void __cudampi__allocAndGatherPowercapRanges(void);
void __cudampi__initDevicePowercapConfig(void);
void __cudampi__applyInitialPowercapsForStrategy(void);
void __cudampi__cmaesCleanup(void);
void __cudampi__resetLocalPowercaps(void);

extern powercapStrategy_t __cudampi__powercapStrategy;
#endif // CUDAMPI_POWERCAP_H
