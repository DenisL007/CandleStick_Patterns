from distutils.core import setup

setup(
    name='CandleStick_Patterns',
    version='0.1.0',
    author=['Den. Random Hacker', 'Sam. Random Hacker']
    author_email=['den@example.com','sam@example.com']
    packages=['candlestick_patterns', 'candlestick_patterns.test'],
    scripts=['bin/stowe-candlestick_patterns.py','bin/wash-candlestick_patterns.py'],
    url='http://pypi.python.org/pypi/CandleStick_Patterns/',
    license='LICENSE.txt',
    description='Useful candlestick_patterns-related stuff.',
    long_description=open('README.txt').read(),
    install_requires=[
        "Django >= 1.1.1",
        "caldav == 0.1.4",
    ],
)