#!/bin/bash
wget -O clothing.zip https://codeload.github.com/bearpaw/clothing-co-parsing/zip/master
unzip clothing.zip
cp -frp clothing-co-parsing-master/* database
rm -rf clothing-co-parsing-master
