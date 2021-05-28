FROM python:3.8

RUN mkdir /data && \
    mkdir /igel

COPY requirements.txt /igel/requirements.txt
RUN pip install -r /igel/requirements.txt

COPY assets /igel/assets
COPY docs /igel/docs
COPY igel /igel/igel
COPY setup.cfg /igel/setup.cfg
COPY setup.py /igel/setup.py
COPY HISTORY.rst /igel/HISTORY.rst
COPY setup.py /igel/setup.py
RUN cd /igel && python setup.py install

VOLUME /data
WORKDIR /data

ENTRYPOINT ["igel"]
CMD ["igel"]
