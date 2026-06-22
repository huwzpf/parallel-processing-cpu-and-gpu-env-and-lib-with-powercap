#ifndef LOGGER_H
#define LOGGER_H

#include <stdarg.h>
#include <stdio.h>
#include <time.h>

#ifdef MPI_LOGGING
#include <mpi.h>
#include <omp.h>
#endif

typedef enum { LOG_DEBUG = 0, LOG_INFO, LOG_WARN, LOG_ERROR } LogLevel;

static const char *LOG_LEVEL_NAMES[] = {"DEBUG", "INFO", "WARN", "ERROR"};

#ifdef ENABLE_LOGGING
#define LOG_LEVEL_THRESHOLD LOG_WARN
#endif

#ifdef ENABLE_LOGGING_DEBUG
#define LOG_LEVEL_THRESHOLD LOG_DEBUG
#endif

#ifdef ENABLE_LOGGING // All logs expect DEBUG in stdout
static void log_message(LogLevel level, const char *format, ...) {
  if (level >= LOG_LEVEL_THRESHOLD) {
    time_t now;
    time(&now);
    struct tm *local = localtime(&now);

    char time_str[20];
    strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", local);

#ifdef MPI_LOGGING
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);

    fprintf(stderr, "[%s] [Thread %d] [%s] [%s] ", processor_name, omp_get_thread_num(), time_str, LOG_LEVEL_NAMES[level]);
#else
    fprintf(stderr, "[%s] [%s] ", time_str, LOG_LEVEL_NAMES[level]);
#endif

    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);

    fprintf(stderr, "\n");
    fflush(stderr);
  }
}
#else
#ifdef ENABLE_LOGGING_DEBUG // All logs in logfile.txt
static void log_message(LogLevel level, const char *format, ...) {
  FILE *log_file = fopen("logfile.txt", "a");
  if (log_file == NULL) {
    fprintf(stderr, "Error opening log file!\n");
    return;
  }

  time_t now;
  time(&now);
  struct tm *local = localtime(&now);

  char time_str[20];
  strftime(time_str, sizeof(time_str), "%Y-%m-%d %H:%M:%S", local);

#ifdef MPI_LOGGING
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  fprintf(log_file, "[%s] [Thread %d] [%s] [%s] ", processor_name, omp_get_thread_num(), time_str, LOG_LEVEL_NAMES[level]);
#else
  fprintf(log_file, "[%s] [%s] ", time_str, LOG_LEVEL_NAMES[level]);
#endif

  va_list args;
  va_start(args, format);
  vfprintf(log_file, format, args);
  va_end(args);

  fprintf(log_file, "\n");

  fclose(log_file);
}
#else  // Logs disabled
static void log_message(LogLevel level, const char *format, ...) {}
#endif // ENABLE_LOGGING_DEBUG
#endif // ENABLE_LOGGING
#endif // LOGGER_H
