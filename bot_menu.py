import logging
import asyncio
from telegram import Update, BotCommand
from telegram.ext import Application, CommandHandler, ContextTypes
from config import TOKEN  # Make sure you have a config.py file with your TOKEN

# Set up logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO)
logger = logging.getLogger(__name__)


async def set_commands(application: Application) -> None:
    """Set bot commands to display in the menu."""
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("menu", "Show main menu"),
        BotCommand("add_cow", "Add a new cow"),
        BotCommand("list_cows", "List your cows"),
        BotCommand("identify", "Identify a cow")
    ]
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands have been set successfully.")


async def main() -> None:
    """Set up the bot menu and exit."""
    # Create the Application and pass it your bot's token.
    application = Application.builder().token(TOKEN).build()

    # Set the commands
    await set_commands(application)

    # Exit the script
    logger.info("Menu creation completed. Exiting the script.")


if __name__ == '__main__':
    asyncio.run(main())
