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
    float global_powercap;       // global power cap in Watts (0 means disabled)
    // Gradient optimisation parameters (used for EDP_GRADIENT_* strategies)
    float start_alpha;           // initial learning rate (default 2.0)
    float alpha_decay;           // learning rate decay per iteration (default 0.99)
    float epsilon_decay;         // epsilon decay factor per iteration (slow->fast schedule)
    float gradient_opt_eps;      // finite-difference / perturbation step (default 5.0)
    unsigned long long cpu_time_window_us; // CPU power cap time window (microseconds)
    // Limit the number of dynamic optimisation updates (0 = unlimited)
    unsigned long long edp_optimization_steps;
} powercap_config_t;

// Loads configuration from a simple key=value file.
// Unknown keys are ignored to keep the parser easily extendable.
// If the file cannot be read, the structure is left with default values.
int load_powercap_config(const char *path, powercap_config_t *config);

#endif // POWERCAP_CONFIG_H
