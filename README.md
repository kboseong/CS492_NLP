본 repo는 NSML 상에서 수행한 korquad-open-ldbd dataset의 학습과정과 inference 과정을 담고 있습니다. 

수업 내 1위를 달성하였습니다.

발표자료는 [https://github.com/kboseong/CS492_NLP/blob/master/nlp_presentation_dingbro.pdf](https://github.com/kboseong/CS492_NLP/blob/master/nlp_presentation_dingbro.pdf) 을 참고해주세요

train.py와 inference.py 실행을 위하여 apex라이브러리를 반드시 설치해주시기 바랍니다.

cuda 10.0, pytorch 1.4 버전에서 프로젝트를 진행하였습니다.

아래의 실행을 통해 train을 시킬 수 있습니다.

```jsx
sh train.sh
```

만들어진 모델이나, aws상에 업로드된 모델을 다운로드 받아 inference.py를 실행시킬 수 있습니다. 해당코드를 실행하면 최종적인 output이 생성됩니다.

```jsx
python inference.py \
		--vocab_file {vocal file direction} \
    --predict_file {the file name of need to predict (json type)} \
    --model_config {model configuration file direction} \
    --model_weight {model weight file direction}

```

vocab file 과 configuration file은 인라이플의 git 홈페이지에서 다운받을 수 있습니다. v1이 학습시킨 모델 입니다.

[https://github.com/enlipleai/kor_pretrain_LM](https://github.com/enlipleai/kor_pretrain_LM)

NSML상에서 best score를 획득한 모델은 aws 상에서 아래 링크를 통해 다운로드할 수 있습니다.

[https://dingbro-garbage.s3.ap-northeast-2.amazonaws.com/nlp_model_final.pt](https://dingbro-garbage.s3.ap-northeast-2.amazonaws.com/nlp_model_final.pt)

# Data augmentation

본 프로젝트에서는 electra generator 를 이용한 data augmentation 을 하였습니다.

자세한 사항은 data_augmentation 주피터 파일을 확인해주시기 바랍니다.
