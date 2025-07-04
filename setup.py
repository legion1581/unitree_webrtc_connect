from setuptools import setup, find_packages

setup(
    name='go2-webrtc-connect',
    version='1.0.0',
    author='legion1581',
    author_email='legion1581@gmail.com',
    packages=find_packages(),
    package_data={
        '': ['*.wasm'],
    },
    include_package_data=True,
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
        'aioice @ git+https://github.com/legion1581/aioice.git@go2'
    ],
)
