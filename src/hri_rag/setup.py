from setuptools import setup, find_packages

setup(
    name='hri_rag',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['setuptools'],
    zip_safe=True,
    author='gali',
    author_email='gali@example.com',
    description='Nodos de RAG y MCP para el sistema HRI',
    license='MIT',
    entry_points={
        'console_scripts': [
            'rag_service_node = hri_rag.rag_service_node:main',
            'mcp_server = hri_rag.mcp_server:main',
        ],
    },
)
