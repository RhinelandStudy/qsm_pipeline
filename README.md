# qsm_pipeline
Pipeline for QSM 3D EPI pre-processing

## Build docker image

```bash

docker build -t qsm_pipeline -f docker/Dockerfile .

```

## Or pull from docker hub

```bash

docker pull dznerheinlandstudie/rheinlandstudie:qsm_pipeline

```

## Run pipeline:

### Using docker

```bash

docker run --rm -v /path/to/inputdata:/input \
                -v /path/to/work:/work \
                -v /path/to//output:/output \
             dznerheinlandstudie/rheinlandstudie:qsm_pipeline \
             run_qsm_pipeline \
                -s /input \
                -w /work \
                -o /output -p 2 -t 2 --cmpth 1

```

The command line options are described briefly if the pipeline is started with only -h option.

### Using Singularity

The pipeline can be run with Singularity by running the singularity image as follows:

```bash


singularity build qsm_pipeline.sif docker://dznerheinlandstudie/rheinlandstudie:qsm_pipeline

```

When the singularit image is created, then it can be run as follows:


```bash

singularity run  -B /path/to/inputdata:/input \
                 -B /path/to/work:/work \
                 -B /path/to/output:/output \
            qsm_pipeline.sif \
            run_qsm_pipeline \ 
                      -s /input \
                      -w /work \
                      -o /output \ 
                      -p 2 -t 2 --cmpth 1
                   
```

