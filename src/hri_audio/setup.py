from setuptools import setup, find_packages

setup(
    name='hri_audio',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['setuptools'],
    zip_safe=True,
    author='gali',
    author_email='gali@example.com',
    description='Nodos de audio para el sistema HRI',
    license='MIT',
    entry_points={
        'console_scripts': [
            'audio_capture_node = hri_audio.audio_capture_node:main',
            'vad_node = hri_audio.vad_node:main',
            'transcription_node = hri_audio.transcription_node:main',
            'voice_identity_node = hri_audio.voice_identity_node:main',
        ],
    },
)
