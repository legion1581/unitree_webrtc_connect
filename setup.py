from setuptools import setup, find_packages

setup(
    name='go2-webrtc-connect',
    version='1.0.0',
    author='legion1581',
    author_email='legion1581@gmail.com',
    packages=find_packages(),
    install_requires=[
        'aiortc==1.9.0', 
        'pycryptodome',
        'opencv-python',
        'sounddevice',
        'pyaudio',
        'requests',
        'wasmtime',
        'flask-socketio',
        'lz4',
        'pydub',
        # Voice Agent Dependencies
        'pipecat-ai[daily,cartesia,openai,silero,deepgram]',
        'python-dotenv',
        'aiohttp',
        'loguru',
        'Pillow',
        'scipy',
        'numpy',
        'weave'
    ],
)
