# -*- coding: utf-8 -*-
"""A nipype pre-processing pipeline for EPI-based QSM data.

Created on Wed Jul 31 2018

@author: stirnbergr

Copyright 2023 Population Health Sciences, German Center for Neurodegenerative Diseases (DZNE)
Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at  http://www.apache.org/licenses/LICENSE-2.0 
Unless required by applicable law or agreed to in writing, software distributed under the 
License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and limitations under the License.

"""

from __future__ import division
import argparse
import sys
import os
import glob
from itertools import chain
from nipype import config, logging

from .QsmEpiPorcessing import create_qsmepiwf

import matplotlib
matplotlib.use('Agg')


def main():
    """
    Command line wrapper for preprocessing data
    """
    parser = argparse.ArgumentParser(
            description='Processing pipeling for 3D-EPI-based R2*/QSM data.',
            epilog='Example: {prog} -s ~/subjects/ --subjects sub01 sub02 '
            '-o ~/output -w ~/work -p 2'.format(
                    prog=os.path.basename(sys.argv[0])),
                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-s', '--scansdir', help='Scans directory where scans'
                        ' for each subjects are already downloaded: '
                        './scans/scan-sub01/QSMEPI*.nii.gz',required=True)

    parser.add_argument('--subjects', help='One or more subject IDs/ '
                        'scansdir-subdirectory names (space separated). '
                        'Each must contain Two *AP.nii.gz and two *PA.nii.gz '
                        'files. If not specified, all subdirectories of '
                        'input subject directory are treated as subject IDs.',
                        default=None, required=False, nargs='+', action='append')
    

    parser.add_argument('-o', '--outputdir',help='Output directory',
                        required=True)
    parser.add_argument('-w', '--workdir', help='Working directory',
                        required=True)
    parser.add_argument('-d', '--debug', help='debug mode',
                        action='store_true')
    parser.add_argument('-p', '--processes', help='parallel processes',
                        default=1, type=int)
    parser.add_argument('-t', '--threads', help='ITK threads',
                        default=1, type=int)

    parser.add_argument('--cmpth', help='comp_mag_phase threads, multiple threads can be specified for'
                                         ' computing magnitude and phase step',default=1,type=int)

    
    args = parser.parse_args()
    
    
    nthreads=args.threads
    cmpth=args.cmpth

    # Create the workflow
    scans_dir  = os.path.abspath(os.path.expandvars(args.scansdir))
    if not os.path.exists(scans_dir):
        raise IOError("Input scans directory does not exist.")

    subject_ids = []

    if args.subjects:
        subject_ids = list(chain.from_iterable(args.subjects))
    else:
        subject_ids = glob.glob(scans_dir.rstrip('/') + '/*')
        subject_ids = [os.path.basename(s.rstrip('/')) for s in subject_ids]



    if not os.path.exists(args.workdir):
        os.makedirs(args.workdir)

    if not os.path.exists(args.outputdir):
        os.makedirs(args.outputdir)


    config.update_config({
        'logging': {'log_directory': args.workdir, 'log_to_file': True},
        'execution': {'job_finished_timeout' : 60,
                      'poll_sleep_duration' : 30,
                      'hash_method' : 'content',
                      'local_hash_check' : False,
                      'stop_on_first_crash':False,
                      'crashdump_dir': args.workdir,
                      'crashfile_format': 'txt'
                       },
                       })

    #config.enable_debug_mode()
    logging.update_logging(config)

    qsmepiwf = create_qsmepiwf(
            data_dir=scans_dir, subj_ids=subject_ids,
            nthreads=nthreads, cmpth=cmpth)

    qsmepiwf.base_dir = args.workdir
    qsmepiwf.inputs.inputnode.outputdir = os.path.abspath(
            os.path.expandvars(args.outputdir)
            )

    # Visualize workflow
    if args.debug:
        qsmepiwf.write_graph(graph2use='colored', simple_form=True)


    # Run workflow
    qsmepiwf.run(plugin='MultiProc', plugin_args={'n_procs' : args.processes})

    print('DONE!!!')


if __name__ == '__main__':
    sys.exit(main())
