package com.qingxiaotuan.simpcore.antihallucination;

import java.util.*;
import java.time.*;

public class KnowledgeModuleGate {
    private Map<String, AccessPolicy> accessPolicies;
    private Map<String, KnowledgeModule> knowledgeModules;
    private Map<String, RoleDefinition> roles;
    private List<AccessAuditLog> auditLogs;
    private int maxAuditLogs;
    private boolean highRiskEnabled;
    
    public KnowledgeModuleGate() {
        this.accessPolicies = new HashMap<>();
        this.knowledgeModules = new HashMap<>();
        this.roles = new HashMap<>();
        this.auditLogs = new ArrayList<>();
        this.maxAuditLogs = 10000;
        this.highRiskEnabled = true;
        
        initDefaultRoles();
        initDefaultModules();
    }
    
    private void initDefaultRoles() {
        RoleDefinition admin = new RoleDefinition("admin", RoleLevel.ADMIN);
        admin.addPermission("read");
        admin.addPermission("write");
        admin.addPermission("manage");
        roles.put("admin", admin);
        
        RoleDefinition expert = new RoleDefinition("expert", RoleLevel.HIGH);
        expert.addPermission("read");
        expert.addPermission("write");
        roles.put("expert", expert);
        
        RoleDefinition analyst = new RoleDefinition("analyst", RoleLevel.MEDIUM);
        analyst.addPermission("read");
        roles.put("analyst", analyst);
        
        RoleDefinition guest = new RoleDefinition("guest", RoleLevel.LOW);
        guest.addPermission("read");
        guest.setReadOnly(true);
        roles.put("guest", guest);
    }
    
    private void initDefaultModules() {
        addKnowledgeModule("security_cve", "CVE漏洞数据库", "security", true);
        addKnowledgeModule("code_audit", "代码审计知识库", "engineering", true);
        addKnowledgeModule("financial_data", "金融数据知识库", "finance", false);
        addKnowledgeModule("medical_reference", "医学参考知识库", "medical", true);
        addKnowledgeModule("general_knowledge", "通用知识库", "general", false);
        
        setAccessPolicy("admin", "security_cve", AccessLevel.FULL);
        setAccessPolicy("expert", "security_cve", AccessLevel.FULL);
        setAccessPolicy("analyst", "security_cve", AccessLevel.READ_ONLY);
        setAccessPolicy("guest", "security_cve", AccessLevel.DENY);
        
        setAccessPolicy("admin", "code_audit", AccessLevel.FULL);
        setAccessPolicy("expert", "code_audit", AccessLevel.FULL);
        setAccessPolicy("analyst", "code_audit", AccessLevel.READ_ONLY);
        setAccessPolicy("guest", "code_audit", AccessLevel.DENY);
        
        setAccessPolicy("admin", "financial_data", AccessLevel.FULL);
        setAccessPolicy("expert", "financial_data", AccessLevel.FULL);
        setAccessPolicy("analyst", "financial_data", AccessLevel.READ_ONLY);
        setAccessPolicy("guest", "financial_data", AccessLevel.READ_ONLY);
        
        setAccessPolicy("admin", "medical_reference", AccessLevel.FULL);
        setAccessPolicy("expert", "medical_reference", AccessLevel.FULL);
        setAccessPolicy("analyst", "medical_reference", AccessLevel.READ_ONLY);
        setAccessPolicy("guest", "medical_reference", AccessLevel.DENY);
        
        setAccessPolicy("admin", "general_knowledge", AccessLevel.FULL);
        setAccessPolicy("expert", "general_knowledge", AccessLevel.FULL);
        setAccessPolicy("analyst", "general_knowledge", AccessLevel.FULL);
        setAccessPolicy("guest", "general_knowledge", AccessLevel.READ_ONLY);
    }
    
    public void addKnowledgeModule(String moduleId, String name, String category, boolean isHighRisk) {
        KnowledgeModule module = new KnowledgeModule(moduleId, name, category, isHighRisk);
        knowledgeModules.put(moduleId, module);
    }
    
    public void setAccessPolicy(String roleId, String moduleId, AccessLevel level) {
        String key = roleId + ":" + moduleId;
        AccessPolicy policy = new AccessPolicy(roleId, moduleId, level);
        accessPolicies.put(key, policy);
    }
    
    public AccessDecision checkAccess(String roleId, String moduleId, String action) {
        AccessDecision decision = new AccessDecision();
        decision.timestamp = LocalDateTime.now();
        decision.roleId = roleId;
        decision.moduleId = moduleId;
        decision.action = action;
        
        RoleDefinition role = roles.get(roleId);
        KnowledgeModule module = knowledgeModules.get(moduleId);
        
        if (role == null) {
            decision.allowed = false;
            decision.reason = "角色不存在: " + roleId;
            logAccess(decision);
            return decision;
        }
        
        if (module == null) {
            decision.allowed = false;
            decision.reason = "知识模块不存在: " + moduleId;
            logAccess(decision);
            return decision;
        }
        
        String policyKey = roleId + ":" + moduleId;
        AccessPolicy policy = accessPolicies.get(policyKey);
        
        if (policy == null) {
            decision.allowed = false;
            decision.reason = "未找到访问策略";
            logAccess(decision);
            return decision;
        }
        
        decision.accessLevel = policy.level;
        
        if (module.isHighRisk && !role.isHighLevel()) {
            decision.allowed = false;
            decision.reason = "高风险模块需要高级角色权限";
            decision.highRiskBlock = true;
            logAccess(decision);
            return decision;
        }
        
        if (!role.hasPermission(action)) {
            decision.allowed = false;
            decision.reason = "角色缺少权限: " + action;
            logAccess(decision);
            return decision;
        }
        
        switch (policy.level) {
            case FULL:
                decision.allowed = true;
                decision.reason = "完全访问权限";
                break;
            case READ_ONLY:
                decision.allowed = "read".equals(action);
                decision.reason = decision.allowed ? "只读访问权限" : "只读模块禁止写操作";
                break;
            case DENY:
                decision.allowed = false;
                decision.reason = "访问被拒绝";
                break;
        }
        
        logAccess(decision);
        return decision;
    }
    
    public void updateAccessPolicy(String roleId, String moduleId, AccessLevel newLevel) {
        setAccessPolicy(roleId, moduleId, newLevel);
        
        RoleDefinition role = roles.get(roleId);
        if (role != null) {
            if (newLevel == AccessLevel.FULL) {
                role.addPermission("read");
                role.addPermission("write");
                role.addPermission("manage");
                role.readOnly = false;
            } else if (newLevel == AccessLevel.READ_ONLY) {
                role.addPermission("read");
                role.readOnly = true;
            }
        }
        
        AccessAuditLog log = new AccessAuditLog();
        log.timestamp = LocalDateTime.now();
        log.roleId = roleId;
        log.moduleId = moduleId;
        log.action = "POLICY_UPDATE";
        log.allowed = true;
        log.reason = "策略更新: " + newLevel;
        auditLogs.add(log);
        trimAuditLogs();
    }
    
    public void addRole(String roleId, RoleLevel level) {
        RoleDefinition role = new RoleDefinition(roleId, level);
        roles.put(roleId, role);
    }
    
    public void addRolePermission(String roleId, String permission) {
        RoleDefinition role = roles.get(roleId);
        if (role != null) {
            role.addPermission(permission);
        }
    }
    
    public List<AccessAuditLog> getAuditLogs(String moduleId) {
        List<AccessAuditLog> filtered = new ArrayList<>();
        for (AccessAuditLog log : auditLogs) {
            if (moduleId == null || moduleId.equals(log.moduleId)) {
                filtered.add(log);
            }
        }
        return filtered;
    }
    
    public List<AccessAuditLog> getRecentAuditLogs(int limit) {
        int start = Math.max(0, auditLogs.size() - limit);
        return new ArrayList<>(auditLogs.subList(start, auditLogs.size()));
    }
    
    private void logAccess(AccessDecision decision) {
        AccessAuditLog log = new AccessAuditLog();
        log.timestamp = decision.timestamp;
        log.roleId = decision.roleId;
        log.moduleId = decision.moduleId;
        log.action = decision.action;
        log.allowed = decision.allowed;
        log.reason = decision.reason;
        log.highRiskBlock = decision.highRiskBlock;
        
        auditLogs.add(log);
        trimAuditLogs();
    }
    
    private void trimAuditLogs() {
        while (auditLogs.size() > maxAuditLogs) {
            auditLogs.remove(0);
        }
    }
    
    public enum RoleLevel {
        ADMIN, HIGH, MEDIUM, LOW
    }
    
    public enum AccessLevel {
        FULL, READ_ONLY, DENY
    }
    
    public static class RoleDefinition {
        public String roleId;
        public RoleLevel level;
        public Set<String> permissions;
        public boolean readOnly;
        
        public RoleDefinition(String roleId, RoleLevel level) {
            this.roleId = roleId;
            this.level = level;
            this.permissions = new HashSet<>();
            this.readOnly = false;
        }
        
        public void addPermission(String permission) {
            permissions.add(permission);
        }
        
        public boolean hasPermission(String permission) {
            return permissions.contains(permission);
        }
        
        public boolean isHighLevel() {
            return level == RoleLevel.ADMIN || level == RoleLevel.HIGH;
        }
        
        public void setReadOnly(boolean readOnly) {
            this.readOnly = readOnly;
        }
    }
    
    public static class KnowledgeModule {
        public String moduleId;
        public String name;
        public String category;
        public boolean isHighRisk;
        
        public KnowledgeModule(String moduleId, String name, String category, boolean isHighRisk) {
            this.moduleId = moduleId;
            this.name = name;
            this.category = category;
            this.isHighRisk = isHighRisk;
        }
    }
    
    public static class AccessPolicy {
        public String roleId;
        public String moduleId;
        public AccessLevel level;
        
        public AccessPolicy(String roleId, String moduleId, AccessLevel level) {
            this.roleId = roleId;
            this.moduleId = moduleId;
            this.level = level;
        }
    }
    
    public static class AccessDecision {
        public LocalDateTime timestamp;
        public String roleId;
        public String moduleId;
        public String action;
        public boolean allowed;
        public String reason;
        public AccessLevel accessLevel;
        public boolean highRiskBlock;
    }
    
    public static class AccessAuditLog {
        public LocalDateTime timestamp;
        public String roleId;
        public String moduleId;
        public String action;
        public boolean allowed;
        public String reason;
        public boolean highRiskBlock;
        
        public String toString() {
            return String.format("[%s] %s -> %s (%s): %s [%s]",
                timestamp.toString(), roleId, moduleId, action,
                allowed ? "ALLOWED" : "DENIED", reason);
        }
    }
}
