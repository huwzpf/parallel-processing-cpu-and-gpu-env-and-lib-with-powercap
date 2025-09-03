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
    config->strategy = BINARY_GREEDY;
    config->cpu_min_powercap = 0.0f;
    config->gpu_min_powercap = 0.0f;
    config->start_powercap = START_POWERCAP_FOR_GRAD_OPT;

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
                if (strcmp(value, "CONTINOUS_EQUAL") == 0) {
                    config->strategy = CONTINOUS_EQUAL;
                } else if (strcmp(value, "BINARY_GREEDY") == 0) {
                    config->strategy = BINARY_GREEDY;
                } else if (strcmp(value, "EDP_GRADIENT_OPT") == 0) {
                    config->strategy = EDP_GRADIENT_OPT;
                }
            } else if (strcmp(key, "cpu_min_powercap") == 0) {
                config->cpu_min_powercap = (float)atof(value);
            } else if (strcmp(key, "gpu_min_powercap") == 0) {
                config->gpu_min_powercap = (float)atof(value);
            } else if (strcmp(key, "start_powercap") == 0) {
                config->start_powercap = (float)atof(value);
            }
            // Unknown keys are ignored to keep parser extendable
        }
    }
    fclose(f);
    return 0;
}

