import pydicom
import os
import numpy as np

# Path to DICOM files
dicom_path = "/Users/hydra/Downloads/288_DICOM"

# Get all DICOM files
files = [f for f in os.listdir(dicom_path) if f.endswith('.dcm')]
files.sort()

print(f"Total DICOM files: {len(files)}")

# Analyze first 10 files
print("\nAnalyzing first 10 files:")
for i, f in enumerate(files[:10]):
    ds = pydicom.dcmread(os.path.join(dicom_path, f))
    print(f"{i+1}. {f}:")
    print(f"   SliceLocation: {ds.get('SliceLocation', 'N/A')}")
    print(f"   InstanceNumber: {ds.get('InstanceNumber', 'N/A')}")
    print(f"   ImagePosition: {ds.get('ImagePosition', 'N/A')}")
    print(f"   ImageSize: {ds.pixel_array.shape if hasattr(ds, 'pixel_array') else 'N/A'}")
    print()

# Check if these are different slices from same patient
print("Checking slice information for all files...")
slice_locations = []
instance_numbers = []

for f in files:
    ds = pydicom.dcmread(os.path.join(dicom_path, f))
    slice_locations.append(ds.get('SliceLocation', None))
    instance_numbers.append(ds.get('InstanceNumber', None))

unique_slices = set([s for s in slice_locations if s is not None])
unique_instances = set([i for i in instance_numbers if i is not None])

print(f"Unique slice locations: {len(unique_slices)}")
print(f"Unique instance numbers: {len(unique_instances)}")

if len(unique_slices) > 1:
    print("These appear to be different slices from the same patient")
elif len(unique_instances) > 1:
    print("These appear to be different instances from the same patient")
else:
    print("These might be the same slice or lack proper metadata")

# Check image dimensions
print("\nChecking image dimensions...")
dimensions = set()
for f in files[:5]:  # Check first 5 files
    ds = pydicom.dcmread(os.path.join(dicom_path, f))
    if hasattr(ds, 'pixel_array'):
        dimensions.add(ds.pixel_array.shape)

print(f"Image dimensions found: {dimensions}")

# Check if these are different patients or same patient different timepoints
print("\nChecking patient and study information...")
patient_ids = set()
study_dates = set()
series_descriptions = set()

for f in files[:20]:  # Check first 20 files
    ds = pydicom.dcmread(os.path.join(dicom_path, f))
    patient_ids.add(ds.get('PatientID', 'N/A'))
    study_dates.add(ds.get('StudyDate', 'N/A'))
    series_descriptions.add(ds.get('SeriesDescription', 'N/A'))

print(f"Unique patient IDs: {patient_ids}")
print(f"Unique study dates: {study_dates}")
print(f"Unique series descriptions: {series_descriptions}") 