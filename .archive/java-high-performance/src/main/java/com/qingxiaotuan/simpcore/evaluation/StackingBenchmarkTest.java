package com.qingxiaotuan.simpcore.evaluation;

import com.qingxiaotuan.simpcore.stacking.*;
import com.qingxiaotuan.simpcore.tokenizer.ChineseBPETokenizer;

import java.util.*;

public class StackingBenchmarkTest {
    
    public static void main(String[] args) {
        System.out.println("==========================================");
        System.out.println("      青小团-7B 参数堆叠性能对比测试");
        System.out.println