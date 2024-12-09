# install pytorch (>1.8.0)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia -y

# install pip libraries
pip install tqdm\
            pillow\
            opencv-python\
            scikit-image==0.18.2\
            scikit-learn==0.24.2\
            scipy==1.7.1\
            joblib\
            numpy\
            torchmetrics\
            ninja\
            cython -i https://pypi.tuna.tsinghua.edu.cn/simple
