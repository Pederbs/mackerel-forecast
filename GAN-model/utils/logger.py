from loguru import logger
import sys

# Remove default handler to customize the format
logger.remove()

# Add a new handler with the default Loguru colors,
# but output the absolute file path and line number.
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{file.path}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True
)

if __name__ == '__main__':
    logger.info("this is a test")
    logger.debug("this is a debug test")
    logger.warning("this is a warning test")
    logger.error("this is an error test")
