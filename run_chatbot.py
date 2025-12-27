import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
print('Loading libraries')
from src.generators.chatbot import DialogBot


def main():
    print('Loading chatbot model')
    bot = DialogBot()
    bot.load('model')

    print('Start a dialogue (ctrl+c for exit):')
    while True:
        try:
            user_input = input('> ').strip()

            if not user_input:
                continue

            response = bot.generate_response(user_input)
            print(f'>> {response}\n')

        except (KeyboardInterrupt, EOFError):
            break


if __name__ == '__main__':
    main()
