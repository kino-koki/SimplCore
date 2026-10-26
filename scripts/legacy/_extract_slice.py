import sys, zipfile
wheel, sp, start, end = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
z = zipfile.ZipFile(wheel)
names = z.namelist()
for n in names[start:end]:
    z.extract(n, sp)
print(f"extracted {start}:{end} -> {len(names[start:end])} files")
