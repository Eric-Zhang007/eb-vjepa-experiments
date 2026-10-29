"""Project-local robosuite environment override: no shared /tmp log writes."""
import robosuite.macros as macros

macros.FILE_LOGGING_LEVEL = None
