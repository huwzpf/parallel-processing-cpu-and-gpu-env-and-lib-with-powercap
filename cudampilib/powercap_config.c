#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>

#include "powercap_config.h"

static void trim(char *str) {
    char *end;
    while (isspace((unsigned char)*str)) str++;
    if (*str == 0) return;
    end = str + strlen(str) - 1;
    while (end > str && isspace((unsigned char)*end)) end--;
    end[1] = '\0';
}

int load_powercap_config(const char *path, powercap_config_t *config) {
    if (!config) return -1;

    // Default values
    config->strategy = DISABLED;
    config->cpu_min_powercap = 0.0;
    config->gpu_min_powercap = 0.0;
    config->start_powercap = 0.5;
    config->global_powercap = 0.0f; // disabled by default
    config->start_alpha = 2.0f;     // default START_ALPHA
    config->alpha_decay = 0.99f;    // default ALPHA_DECAY
    config->epsilon_decay = 0.99f;  // default EPSILON_DECAY
    config->gradient_opt_eps = 5.0f;// default GRADIENT_OPT_EPS
    // Default CPU time window: 1 second (microseconds)
    config->cpu_time_window_us = 1000000ULL;

    FILE *f = fopen(path, "r");
    if (!f) {
        return -1; // silently ignore missing file
    }

    char line[256];
    while (fgets(line, sizeof(line), f)) {
        char *comment = strchr(line, '#');
        if (comment) *comment = '\0';
        trim(line);
        if (strlen(line) == 0) continue;

        char key[128];
        char value[128];
        if (sscanf(line, "%127[^=]=%127s", key, value) == 2) {
            if (strcmp(key, "strategy") == 0) {
                if (strcmp(value, "DISABLED") == 0) {
                    config->strategy = DISABLED;
                } else if (strcmp(value, "CONTINOUS_EQUAL") == 0) {
                    config->strategy = CONTINOUS_EQUAL;
                } else if (strcmp(value, "EQUAL_SHARE_CONTINOUS_EQUAL") == 0) {
                    config->strategy = EQUAL_SHARE_CONTINOUS_EQUAL;
                } else if (strcmp(value, "EQUAL_SPLIT") == 0) {
                    config->strategy = EQUAL_SPLIT;
                } else if (strcmp(value, "BINARY_GREEDY") == 0) {
                    config->strategy = BINARY_GREEDY;
                } else if (strcmp(value, "EQUAL_SHARE_BINARY_GREEDY") == 0) {
                    config->strategy = EQUAL_SHARE_BINARY_GREEDY;
                } else if (strcmp(value, "EDP_GRADIENT_SIMPLE") == 0) {
                    config->strategy = EDP_GRADIENT_SIMPLE;
                } else if (strcmp(value, "EDP_GRADIENT_SPSA") == 0) {
                    config->strategy = EDP_GRADIENT_SPSA;
                } else if (strcmp(value, "EDP_GRADIENT_SIMPLE_ADAPTIVE") == 0) {
                    config->strategy = EDP_GRADIENT_SIMPLE_ADAPTIVE;
                } else if (strcmp(value, "EDP_GRADIENT_CMAES") == 0) {
                    config->strategy = EDP_GRADIENT_CMAES;
                }
            } else if (strcmp(key, "cpu_min_powercap") == 0) {
                config->cpu_min_powercap = (float)atof(value);
            } else if (strcmp(key, "gpu_min_powercap") == 0) {
                config->gpu_min_powercap = (float)atof(value);
            } else if (strcmp(key, "start_powercap") == 0) {
                config->start_powercap = (float)atof(value);
            } else if (strcmp(key, "start_alpha") == 0) {
                config->start_alpha = (float)atof(value);
            } else if (strcmp(key, "alpha_decay") == 0) {
                config->alpha_decay = (float)atof(value);
            } else if (strcmp(key, "epsilon_decay") == 0) {
                config->epsilon_decay = (float)atof(value);
            } else if (strcmp(key, "gradient_opt_eps") == 0) {
                config->gradient_opt_eps = (float)atof(value);
            } else if (strcmp(key, "global_powercap") == 0) {
                config->global_powercap = (float)atof(value);
            } else if (strcmp(key, "cpu_time_window_us") == 0) {
                // microseconds
                config->cpu_time_window_us = (unsigned long long)strtoull(value, NULL, 10);
            }
            // Unknown keys are ignored to keep parser extendable
        }
    }
    fclose(f);
    return 0;
}
