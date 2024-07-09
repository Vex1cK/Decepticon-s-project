from oxr import getFromDotenv
from model import get_model
import logging
from aiogram import Bot, Dispatcher, types
from aiogram.utils import executor
import requests
import librosa
import io
import torch.nn.functional as F
import torch
from random import choice

# Установите уровень логирования
logging.basicConfig(level=logging.INFO)

# Инициализация бота и диспетчера
API_TOKEN = getFromDotenv('BOT_API')
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = get_model().to(DEVICE)

logging.info("ready")

# Обработчик команды /start
@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Отправь мне голосовое, и я попробую понять картавишь ты или нет =)")

# Обработчик голосового сообщения
@dp.message_handler(content_types=[types.ContentType.VOICE])
async def handle_voice(message: types.Message):
    
    try:
        file_id = message.voice.file_id
        file_info = await bot.get_file(file_id)
        file_path = file_info.file_path
        
        # Скачиваем файл
        response = requests.get(f'https://api.telegram.org/file/bot{API_TOKEN}/{file_path}')
        
        # Работаем с файлом в памяти
        audio_bytes = io.BytesIO(response.content)
    except Exception:
        await message.answer("У меня произошла какая-то странная ошибка =(\nПопробуй еще раз")

    await message.answer("Обрабатываю...")

    try:
    
        # Преобразуем ogg файл в wav с помощью soundfile
        y, sr = librosa.load(audio_bytes, sr=None)

        mfccs = torch.tensor(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13))
        logging.info(f"Got mfccs. shape: {mfccs.shape}")
        

        if mfccs.shape[1] > 619:
            chunk_size = 619
            last_chunk_size = mfccs.size(1) % chunk_size

            # Разделяем тензор на кусочки
            chunks = torch.split(mfccs, chunk_size, dim=1)

            # Если последний кусок меньше требуемого размера, дополним его
            if last_chunk_size > 0:
                last_chunk = chunks[-1]
                padding_size = chunk_size - last_chunk_size
                padded_last_chunk = F.pad(last_chunk, (0, padding_size))
                chunks = chunks[:-1] + (padded_last_chunk,)
            
            chunks = torch.stack(chunks)

        else:
            pad_size = 619 - mfccs.shape[1]
            chunks = F.pad(mfccs, (0, pad_size), "constant", 0).unsqueeze(0)
        
        logging.info(f"chunks shape: {chunks.shape}")
        logging.info("evaluating")
        with torch.no_grad():
            chunks = chunks.to(DEVICE)
            predicts = model(chunks).squeeze(-1)
            predicts = list(map(lambda x: int(x > 0.0), predicts))
            predicts = any(predicts)
        
        await message.reply(choice([f'Я думаю ты картавишь', "Мне кажется ты картавишь", "Кажется, ты картавишь"]) \
                             if predicts else \
                             choice([f'Я не думаю что ты картавишь', "Ты вряд-ли картавишь", "Мне кажется, что ты не картавишь"]))
        await message.answer("Можешь отправить еще голосовых)")
    
    except Exception as ex:
        logging.error(ex)
        await message.reply("Упс! У меня произошла какая-то ошибка... Попробуй еще раз позже! =)")
    


if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
