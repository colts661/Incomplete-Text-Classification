### Note

Raw data files are stored in this directory. We provided convenient code snippets to download and place data files for the 3 datasets we experimented. Please run the commands in the **repository root directory**.

#### DBPedia-small
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WuSJyk-O8HzJ-eb4I2l6uH6A49fdWCZU' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WuSJyk-O8HzJ-eb4I2l6uH6A49fdWCZU" -O 'data/raw/DBPedia-small.zip' && rm -rf /tmp/cookies.txt
mkdir data/processed/DBPedia-small
cd data/raw
unzip -o DBPedia-small.zip
rm DBPedia-small.zip
cd ../..
```

#### DBPedia
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1dJPbJZCDYGYgSe8vzWRBB2UjHbDblvcf' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1dJPbJZCDYGYgSe8vzWRBB2UjHbDblvcf" -O 'data/raw/DBPedia.zip' && rm -rf /tmp/cookies.txt
mkdir data/processed/DBPedia
cd data/raw
unzip -o DBPedia.zip
rm DBPedia.zip
cd ../..
```

#### nyt-fine
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1MNzwxP3zlYfZ8_Ni6r5wy03AK5BDgrWT' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1MNzwxP3zlYfZ8_Ni6r5wy03AK5BDgrWT" -O 'data/raw/nyt-fine.zip' && rm -rf /tmp/cookies.txt
mkdir data/processed/nyt-fine
cd data/raw
unzip -o nyt-fine.zip
rm nyt-fine.zip
cd ../..
```

#### Reddit
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-awMLNcNm32OV6NlaUpcGeR0_8hk7nR5' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-awMLNcNm32OV6NlaUpcGeR0_8hk7nR5" -O 'data/raw/Reddit.zip' && rm -rf /tmp/cookies.txt
mkdir data/processed/Reddit
cd data/raw
unzip -o Reddit.zip
rm Reddit.zip
cd ../..
```
