import os
import numpy as np
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand, MenuButton, MenuButtonCommands
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, filters, ContextTypes
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.layers import DepthwiseConv2D
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image
import io
import json
import logging
from config import TOKEN, CONFIDENCE_THRESHOLD

# Set up logging
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Custom object to handle the 'groups' parameter
class CustomDepthwiseConv2D(DepthwiseConv2D):
    def __init__(self, *args, **kwargs):
        if 'groups' in kwargs:
            del kwargs['groups']
        super().__init__(*args, **kwargs)

# Load the trained model with custom objects
model = load_model('cow_recognition_model.h5', custom_objects={'DepthwiseConv2D': CustomDepthwiseConv2D})

# Load the pre-trained MobileNetV2 model for cow detection
cow_detector = MobileNetV2(weights='imagenet')

# User and cow data
user_data = {}
cow_data = {}
missing_cows = []

# Load data from file
def load_data():
    global user_data, cow_data, missing_cows
    if os.path.exists('user_data.json'):
        with open('user_data.json', 'r') as f:
            user_data = json.load(f)
    if os.path.exists('cow_data.json'):
        with open('cow_data.json', 'r') as f:
            cow_data = json.load(f)
    if os.path.exists('missing_cows.json'):
        with open('missing_cows.json', 'r') as f:
            missing_cows = json.load(f)
    logger.debug(f"Loaded data: user_data={user_data}, cow_data={cow_data}, missing_cows={missing_cows}")

# Save data to file
def save_data():
    with open('user_data.json', 'w') as f:
        json.dump(user_data, f)
    with open('cow_data.json', 'w') as f:
        json.dump(cow_data, f)
    with open('missing_cows.json', 'w') as f:
        json.dump(missing_cows, f)
    logger.debug(f"Saved data: user_data={user_data}, cow_data={cow_data}, missing_cows={missing_cows}")

# Function to preprocess the image
def preprocess_image(image):
    img = Image.open(io.BytesIO(image)).convert('RGB')
    img = img.resize((224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

async def set_commands(application: Application):
    commands = [
        BotCommand("start", "Start the bot"),
        BotCommand("menu", "Show main menu"),
        BotCommand("add_cow", "Add a new cow"),
        BotCommand("list_cows", "List your cows"),
        BotCommand("identify", "Identify a cow")
    ]
    await application.bot.set_my_commands(commands)

async def set_menu(application: Application):
    await application.bot.set_chat_menu_button(menu_button=MenuButtonCommands())

# Command handler for /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    user_name = update.effective_user.full_name
    if user_id not in user_data:
        user_data[user_id] = {"name": user_name, "cows": []}
    else:
        user_data[user_id]["name"] = user_name
    save_data()
    
    await show_main_menu(update, context)

    # Set commands and menu for the user
    await set_commands(context.application)
    await set_menu(context.application)

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = [
        [InlineKeyboardButton("🐮 Add Cow", callback_data='add_cow')],
        [InlineKeyboardButton("📋 My Cows", callback_data='list_cows')],
        [InlineKeyboardButton("🔍 Identify Cow", callback_data='identify_cow')]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    await update.effective_message.reply_text('What would you like to do?', reply_markup=reply_markup)

async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await show_main_menu(update, context)

async def add_cow_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data['awaiting_cow_name'] = True
    await update.message.reply_text("Please enter the name of your cow. 🐄")

async def identify_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text("Please send a photo of the cow you want to identify. 🔍")

# Callback query handler
async def button(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    
    if query.data == 'add_cow':
        context.user_data['awaiting_cow_name'] = True
        await query.message.reply_text("Please enter the name of your cow. 🐄")
    elif query.data == 'list_cows':
        await list_cows(update, context)
    elif query.data == 'identify_cow':
        await query.message.reply_text("Please send a photo of the cow you want to identify. 🔍")
    elif query.data.startswith('remove_cow_'):
        cow_id = query.data.split('_')[-1]
        await remove_cow(update, context, cow_id)
    elif query.data.startswith('mark_missing_'):
        cow_id = query.data.split('_')[-1]
        await mark_missing(update, context, cow_id)

# Function to handle photos
async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    photo_file = await update.message.photo[-1].get_file()
    photo = await photo_file.download_as_bytearray()
    
    # Preprocess the image for MobileNetV2
    img_array = preprocess_image(photo)
    
    # Use MobileNetV2 to detect if it's a cow
    predictions = cow_detector.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]
    
    is_cow = any(pred[1] == 'ox' or pred[1] == 'cow' for pred in decoded_predictions)
    
    if not is_cow:
        await update.message.reply_text("This doesn't appear to be a cow. Please send a photo of a cow. 🚫🐮")
        return
    
    # If it's a cow, proceed with cow recognition model
    prediction = model.predict(img_array)
    cow_id = str(np.argmax(prediction))
    confidence = np.max(prediction)
    
    if 'adding_cow' in context.user_data:
        # User is adding a new cow
        cow_name = context.user_data['adding_cow']
        cow_data[cow_id] = {"name": cow_name, "owner": user_id, "photo": photo_file.file_id}
        if cow_id not in user_data[user_id]["cows"]:
            user_data[user_id]["cows"].append(cow_id)
        save_data()
        del context.user_data['adding_cow']
        await update.message.reply_text(f"Cow {cow_name} has been added successfully! 🎉")
    else:
        # User is identifying a cow
        if cow_id in cow_data:
            cow_info = cow_data[cow_id]
            owner_name = user_data[cow_info["owner"]]["name"]
            await update.message.reply_text(f"Cow identified with {confidence:.2%} confidence!\nName: {cow_info['name']}\nOwner: {owner_name}")
        else:
            await update.message.reply_text(f"This cow is not in our database. Would you like to add it? 🆕")

# Function to list user's cows
async def list_cows(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    logger.debug(f"Listing cows for user {user_id}")
    logger.debug(f"User data: {user_data}")
    logger.debug(f"Cow data: {cow_data}")
    
    if user_id not in user_data or not user_data[user_id]["cows"]:
        await update.effective_message.reply_text("You don't have any cows registered. 😢")
        return
    
    # Use a set to ensure unique cow IDs
    unique_cow_ids = set(user_data[user_id]["cows"])
    
    for cow_id in unique_cow_ids:
        logger.debug(f"Processing cow {cow_id}")
        if cow_id not in cow_data:
            logger.error(f"Cow {cow_id} not found in cow_data")
            continue
        cow_info = cow_data[cow_id]
        keyboard = [
            [InlineKeyboardButton("🗑️ Remove", callback_data=f'remove_cow_{cow_id}'),
             InlineKeyboardButton("🚨 Mark as Missing", callback_data=f'mark_missing_{cow_id}')]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=cow_info["photo"], 
                                     caption=f"Name: {cow_info['name']}", reply_markup=reply_markup)

    # Update user_data to remove duplicates
    user_data[user_id]["cows"] = list(unique_cow_ids)
    save_data()

# Function to remove a cow
async def remove_cow(update: Update, context: ContextTypes.DEFAULT_TYPE, cow_id: str) -> None:
    user_id = str(update.effective_user.id)
    if cow_id in user_data[user_id]["cows"]:
        user_data[user_id]["cows"].remove(cow_id)
        del cow_data[cow_id]
        save_data()
        await update.callback_query.message.reply_text(f"Cow has been removed successfully. 👋")
    else:
        await update.callback_query.message.reply_text("This cow doesn't belong to you or doesn't exist. 🚫")

# Function to mark a cow as missing
async def mark_missing(update: Update, context: ContextTypes.DEFAULT_TYPE, cow_id: str) -> None:
    user_id = str(update.effective_user.id)
    if cow_id in user_data[user_id]["cows"]:
        missing_cows.append(cow_id)
        save_data()
        cow_info = cow_data[cow_id]
        # Notify all users
        for user in user_data:
            await context.bot.send_photo(chat_id=user, photo=cow_info["photo"], 
                                         caption=f"🚨 MISSING COW ALERT 🚨\nName: {cow_info['name']}\nOwner: {user_data[user_id]['name']}\nPlease contact the owner if found.")
    else:
        await update.callback_query.message.reply_text("This cow doesn't belong to you or doesn't exist. 🚫")

# Function to handle text messages
async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_id = str(update.effective_user.id)
    if 'awaiting_cow_name' in context.user_data:
        context.user_data['adding_cow'] = update.message.text
        del context.user_data['awaiting_cow_name']
        await update.message.reply_text("Great! Now please send a photo of your cow. 📸")
    else:
        await update.message.reply_text("I'm sorry, I didn't understand that. Please use the menu options or commands. 🤔")

# Set up the application
def main() -> None:
    load_data()
    application = Application.builder().token(TOKEN).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("menu", menu))
    application.add_handler(CommandHandler("add_cow", add_cow_command))
    application.add_handler(CommandHandler("list_cows", list_cows))
    application.add_handler(CommandHandler("identify", identify_command))
    application.add_handler(CallbackQueryHandler(button))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    application.run_polling()

if __name__ == '__main__':
    main()
