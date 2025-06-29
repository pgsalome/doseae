import pycurtv2
import glob
from os.path import dirname
import os
from pycurtv2.converters.dicom import DicomConverter
g = "/data/pgsal/data/TCIA_Subjects1/NSCLC-Cetuximab_wdir1/workflows_output/0617720595/20000909/TRA/NS/RTDOSE-PLANf16T-PHY-MPT/RTDOSEFX1HETERO_PN_69455531"
ref_ct = [x for x in glob.glob(dirname(dirname(dirname(dirname(g)))) + '/*/*/*CT-*MP1*/*') if 'RTSTRUCT' not in x and os.path.isdir(x)]
print(ref_ct)
dd = DicomConverter(toConvert=ref_ct[0],convert_to="nrrd")
dd.convert_ps()
dd = DicomConverter(toConvert=g,convert_to="nrrd")
dd.convert_ps()

