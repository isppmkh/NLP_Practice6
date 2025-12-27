import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print('Loading libraries')

import kagglehub
import pandas as pd

from src.generators.chatbot import DialogBot


def main():
    print('Downloading DailyDialog dataset')

    os.makedirs('train_data', exist_ok=True)
    os.environ['KAGGLEHUB_CACHE'] = './train_data'

    path = kagglehub.dataset_download(
        'thedevastator/dailydialog-unlock-the-conversation-potential-in'
    )

    print('Loading and cleaning data')
    df = pd.read_csv(f'{path}/train.csv')

    df['dialog'] = (
        df['dialog']
        .astype(str)
        .str.replace(r'[\[\]\'\']', '', regex=True)
        .str.replace(r'\\n', ' ', regex=True)
        .str.replace(r'\s+', ' ', regex=True)
        .str.strip()
    )

    text = ' '.join(df['dialog'].tolist())

    print('Starting training')

    bot = DialogBot()
    bot.train(text)
    bot.save()

    print('Training complete')


if __name__ == '__main__':
    main()

