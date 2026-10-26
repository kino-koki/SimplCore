# -*- coding: utf-8 -*-
import ctypes
from ctypes import wintypes

class PMC(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]

def methodA():
    import ctypes
    from ctypes import wintypes
    Kernel32 = ctypes.WinDLL('Kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    GetCurrentProcess = Kernel32.GetCurrentProcess
    GetCurrentProcess.restype = wintypes.HANDLE
    gpm = psapi.GetProcessMemoryInfo
    gpm.argtypes = (wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD)
    gpm.restype = ctypes.c_long
    pm = PMC(); pm.cb = ctypes.sizeof(pm)
    r = gpm(GetCurrentProcess(), ctypes.byref(pm), ctypes.sizeof(pm))
    return r, pm.WorkingSetSize // 1024, pm.cb, ctypes.get_last_error()

def methodB_python():
    # 用内建 OS psutil 等价: 无 psutil 时退回 GlobalMemoryStatus 不做; 这里读 status
    stat = {}
    return stat

if __name__ == '__main__':
    print('method A:', methodA())