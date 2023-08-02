CORPUS=$1

mkdir shard/${CORPUS}

split -d -a 4 -n r/2048 corpus/${CORPUS}.txt shard/${CORPUS}/${CORPUS}_
#"-d": 숫자로 표현
#       ex) corpus_aa, corpus_ab, ... ⇒ corpus_00, corpus_01, ...
#"-a": 0을 넣어 자릿수 맞추기
#       ex) corpus_0000, corpus_0001, ...
#"-n r/N": 라운드로빈 방식으로 N개로 분할
