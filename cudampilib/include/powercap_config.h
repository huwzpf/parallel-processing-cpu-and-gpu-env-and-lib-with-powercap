#ifndef POWERCAP_CONFIG_H
#define POWERCAP_CONFIG_H

#include "cudampilib.h"

// Structure holding power capping configuration
// Additional fields can be easily added to this structure
// to extend configuration options.
typedef struct {
    powercapStrategy_t strategy; // selected power capping strategy
    float cpu_min_powercap;      // minimum CPU power cap (for continuous strategy)
    float gpu_min_powercap;      // minimum GPU power cap (for continuous strategy)
    float start_powercap;        // starting power cap (for gradient optimization)
    unsigned long long cpu_time_window_us; // CPU power cap time window (microseconds)
} powercap_config_t;

// Loads configuration from a simple key=value file.
// Unknown keys are ignored to keep the parser easily extendable.
// If the file cannot be read, the structure is left with default values.
int load_powercap_config(const char *path, powercap_config_t *config);

#endif // POWERCAP_CONFIG_H
