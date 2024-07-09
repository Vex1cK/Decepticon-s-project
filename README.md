# HowBurrBot

HowBurrBot is a Transformer-based neural network designed to detect rhotacism (non-standard pronunciation of the 'r' sound) in audio files. The neural network is wrapped in a Telegram bot (@howburrbot). Users can send voice messages to the bot, which then predicts whether the sender has rhotacism.

## Features

- **State-of-the-art Speech Recognition**: Utilizes a Transformer neural network for accurate detection of rhotacism.
- **Easy to Use**: Simply send a voice message to the Telegram bot (@howburrbot) and receive a prediction.
- **Real-time Processing**: Quick response times for analyzing audio files.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**
    ```bash
    git clone https://github.com/yourusername/howburrbot.git
    cd howburrbot
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your Telegram bot:**
    - Create a new bot on Telegram by talking to [@BotFather](https://t.me/BotFather).
    - Copy the bot token and paste it into the `config.py` file.

5. **Run the bot:**
    ```bash
    python bot.py
    ```

## Usage

Once the bot is running, you can interact with it on Telegram:

1. Open Telegram and search for @howburrbot.
2. Send a voice message to the bot.
3. Receive a prediction on whether the sender has rhotacism.

## Contributing

We welcome contributions from the community! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add new feature'`).
4. Push to the branch (`git push origin feature/YourFeature`).
5. Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, please open an issue on GitHub or contact us directly at [your-email@example.com](mailto:your-email@example.com).

---

We hope you find HowBurrBot useful and we look forward to your feedback!
