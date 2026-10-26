import ctypes, os
from ctypes import wintypes
k32 = ctypes.WinDLL("Kernel32", use_last_error=True)
class PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]
ps = ctypes.WinDLL("psapi", use_last_error=True)
h = k32.GetCurrentProcess
h.restype = wintypes.HANDLE
handle = h()
print("handle:", handle, "err", ctypes.get_last_error())
f = ps.GetProcessMemoryInfo
f.argtypes = (wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD)
f.restype = ctypes.c_long
pm = PMC(); pm.cb = ctypes.sizeof(pm)
ok = f(handle, ctypes.byref(pm), ctypes.sizeof(pm))
print("ok:", ok, "err:", ctypes.get_last_error(), "WS:", pm.WorkingSetSize, "Peak:", pm.PeakWorkingSetSize)
buf = ctypes.create_string_buffer(8*1024*1024)
print("after 8MB alloc WS:", end=" ")
pm2 = PMC(); pm2.cb = ctypes.sizeof(pm2)
print(f(handle, ctypes.byref(pm2), ctypes.sizeof(pm2)), pm2.WorkingSetSize)
