FROM centos:7

#----------------------------
# Install common dependencies
#----------------------------
#too much libs for MCR (libmwcoder_types etc ) to work

RUN yum install -y -q epel-release bzip2 ca-certificates curl perl unzip bc dc gcc mono-mcs gcc-c++ opencl-headers libXt libXext libXrender libXtst libX11 libXcomposite libXcursor libXdamage libXfixes libXft libXi libXrandr libXrender libXScrnSaver libXxf86vm alsa-lib atk cairo cups-libs fontconfig GConf2 gtk2 gdk-pixbuf2 gnome-vfs2 gstreamer1-plugins-base gstreamer1 pango libsndfile libxcb libxslt \
    && yum clean packages \
    && rm -rf /var/cache/yum/* /tmp/* /var/tmp/*

# Set paths and evn vars 
#-------------------------------------------------
ENV PATH=/opt:/opt/miniconda/envs/default/bin:/opt/fsl/bin:/usr/local/cuda/bin:/usr/local/nvidia:$PATH \
    FSLDIR=/opt/fsl \
    FSLOUTPUTTYPE='NIFTI_GZ' \
    LD_LIBRARY_PATH=/opt/fsl/lib:/usr/local/cuda/lib64/:/usr/local/nvidia:$LD_LIBRARY_PATH \
    LC_ALL=en_US.UTF-8 \
    LANG=en_US.UTF-8 \
    STARTUP="/opt/startup.sh"

COPY ./src /opt/

RUN echo "Getting FSL binaries..." \
    && yum install -y wget make zlib-devel \
    && wget -q https://fsl.fmrib.ox.ac.uk/fsldownloads/fsl-5.0.10-sources.tar.gz \
    && tar zxf fsl-5.0.10-sources.tar.gz \
    && cd fsl \
    && ./build || true  \
    && cd / \
    && chmod a+x /opt/copy_bin.sh && ./opt/copy_bin.sh \
    && chmod a+x /opt/fsl/bin/* \
    && rm -rf fsl-5.0.10-sources.tar.gz && rm -rf /fsl


#-------------------------------------------------
# Install Miniconda, and set up Python environment
#-------------------------------------------------
RUN echo "Downloading Miniconda installer ..." \
    && curl -sSL -o miniconda.sh http://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && bash miniconda.sh -b -p /opt/miniconda \
    && rm -f miniconda.sh \
    && /opt/miniconda/bin/conda config --add channels conda-forge \
    && /opt/miniconda/bin/conda create -y -n default python=3.6 \
    && source /opt/miniconda/bin/activate default \
    && conda clean -y --all \
    && pip install -U --no-cache-dir pip \
    && cd /opt \
    && pip install -r requirements.txt \
    && pip install -e .


RUN echo "Installing pipeline..." \
    && chmod a+x /opt/miniconda/bin/* \
    && source /opt/miniconda/bin/activate default \
    && mkdir -p ~/.config/matplotlib \
    && echo 'backend: Agg' > ~/.config/matplotlib/matplotlibrc \
    && sed -i 's/backend      : TkAgg/backend      : Agg/' /opt/miniconda/envs/default/lib/python3.6/site-packages/matplotlib/mpl-data/matplotlibrc \
    && if [ ! -f "$STARTUP" ]; then \
       echo '#!/usr/bin/env bash' >> $STARTUP; \
       echo 'echo "startup: setting environment"...' >> $STARTUP; \
       echo 'source  /docker_environment' >> $STARTUP; \
       echo 'source /opt/miniconda/bin/activate default' >> $STARTUP; \
       echo 'export LD_LIBRARY_PATH=/.singularity.d/libs:/lib64:$LD_LIBRARY_PATH' >> $STARTUP; \
       echo 'IFS=";" read -ra COMMANDS <<< "$@"' >> $STARTUP; \
       echo 'for command in "${COMMANDS[@]}";do' >> $STARTUP; \
       echo 'echo "`date` : executing ${command}";' >> $STARTUP; \
       echo '${command}' >> $STARTUP; \
       echo 'done' >> $STARTUP; \
       fi \
    && chmod 755 $STARTUP


ENTRYPOINT ["/opt/startup.sh"]
