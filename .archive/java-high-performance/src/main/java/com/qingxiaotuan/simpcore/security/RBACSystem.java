package com.qingxiaotuan.simpcore.security;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.CopyOnWriteArrayList;

public class RBACSystem {

    private final Map<String, Role> roles = new ConcurrentHashMap<>();
    private final Map<String, User> users = new ConcurrentHashMap<>();
    private final Map<String, Permission> permissions = new ConcurrentHashMap<>();
    private final Map<String, Session> sessions = new ConcurrentHashMap<>();
    private final List<AuditLog> auditLogs = new CopyOnWriteArrayList<>();
    private final long sessionTimeoutMs = 3600000;

    public void definePermission(String name, String resource, String action) {
        permissions.put(name, new Permission(name, resource, action));
    }

    public void createRole(String name, Set<String> permissionNames) {
        Set<Permission> perms = new HashSet<>();
        for (String permName : permissionNames) {
            Permission p = permissions.get(permName);
            if (p != null) {
                perms.add(p);
            }
        }
        roles.put(name, new Role(name, perms));
    }

    public void createUser(String userId, String... roleNames) {
        Set<Role> userRoles = new HashSet<>();
        for (String roleName : roleNames) {
            Role r = roles.get(roleName);
            if (r != null) {
                userRoles.add(r);
            }
        }
        users.put(userId, new User(userId, userRoles));
    }

    public boolean hasPermission(String userId, String resource, String action) {
        User user = users.get(userId);
        if (user == null) {
            return false;
        }
        return user.hasPermission(resource, action);
    }

    public boolean canAccessModel(String userId, String modelId) {
        return hasPermission(userId, "model:" + modelId, "access");
    }

    public boolean canModifyModel(String userId, String modelId) {
        return hasPermission(userId, "model:" + modelId, "modify");
    }

    public boolean canExecuteInference(String userId) {
        return hasPermission(userId, "inference", "execute");
    }

    public Set<String> getUserPermissions(String userId) {
        User user = users.get(userId);
        if (user == null) {
            return Collections.emptySet();
        }
        Set<String> result = new HashSet<>();
        for (Role role : user.getRoles()) {
            for (Permission perm : role.permissions()) {
                result.add(perm.name());
            }
        }
        return result;
    }

    public void removeUser(String userId) {
        users.remove(userId);
    }

    public void updateUserRoles(String userId, String... newRoles) {
        Set<Role> userRoles = new HashSet<>();
        for (String roleName : newRoles) {
            Role r = roles.get(roleName);
            if (r != null) {
                userRoles.add(r);
            }
        }
        users.put(userId, new User(userId, userRoles));
    }

    public static RBACSystem createDefaultSystem() {
        RBACSystem system = new RBACSystem();

        system.definePermission("model_access", "model:*", "access");
        system.definePermission("model_modify", "model:*", "modify");
        system.definePermission("model_create", "model:*", "create");
        system.definePermission("model_delete", "model:*", "delete");
        system.definePermission("inference_execute", "inference", "execute");
        system.definePermission("inference_monitor", "inference", "monitor");
        system.definePermission("training_start", "training", "start");
        system.definePermission("training_stop", "training", "stop");
        system.definePermission("training_monitor", "training", "monitor");
        system.definePermission("config_access", "config:*", "access");
        system.definePermission("config_modify", "config:*", "modify");
        system.definePermission("admin_all", "*", "*");

        system.createRole("admin", Set.of("admin_all"));
        system.createRole("model_engineer", Set.of("model_access", "model_modify", "model_create", "inference_execute", "training_start", "training_stop", "training_monitor"));
        system.createRole("researcher", Set.of("model_access", "inference_execute", "inference_monitor", "training_monitor"));
        system.createRole("user", Set.of("inference_execute"));
        system.createRole("monitor", Set.of("inference_monitor", "training_monitor", "config_access"));

        return system;
    }
    
    public String createSession(String userId) {
        User user = users.get(userId);
        if (user == null) {
            return null;
        }
        String sessionId = UUID.randomUUID().toString();
        Session session = new Session(sessionId, userId, System.currentTimeMillis());
        sessions.put(sessionId, session);
        logAudit(userId, "session", "create", "Session created");
        return sessionId;
    }
    
    public boolean validateSession(String sessionId) {
        Session session = sessions.get(sessionId);
        if (session == null) {
            return false;
        }
        if (System.currentTimeMillis() - session.createdAt > sessionTimeoutMs) {
            sessions.remove(sessionId);
            return false;
        }
        return true;
    }
    
    public String getUserIdFromSession(String sessionId) {
        Session session = sessions.get(sessionId);
        return session != null ? session.userId : null;
    }
    
    public void invalidateSession(String sessionId) {
        Session session = sessions.remove(sessionId);
        if (session != null) {
            logAudit(session.userId, "session", "invalidate", "Session invalidated");
        }
    }
    
    public void invalidateAllUserSessions(String userId) {
        sessions.entrySet().removeIf(entry -> {
            if (entry.getValue().userId.equals(userId)) {
                logAudit(userId, "session", "invalidate_all", "All sessions invalidated");
                return true;
            }
            return false;
        });
    }
    
    public boolean hasPermissionWithSession(String sessionId, String resource, String action) {
        String userId = getUserIdFromSession(sessionId);
        if (userId == null) {
            return false;
        }
        if (!validateSession(sessionId)) {
            return false;
        }
        boolean result = hasPermission(userId, resource, action);
        logAudit(userId, resource, action, result ? "Allowed" : "Denied");
        return result;
    }
    
    private void logAudit(String userId, String resource, String action, String detail) {
        AuditLog log = new AuditLog(
            System.currentTimeMillis(),
            userId,
            resource,
            action,
            detail
        );
        auditLogs.add(log);
        if (auditLogs.size() > 10000) {
            auditLogs.remove(0);
        }
    }
    
    public List<AuditLog> getAuditLogs(long startTime, long endTime) {
        List<AuditLog> result = new ArrayList<>();
        for (AuditLog log : auditLogs) {
            if (log.timestamp >= startTime && log.timestamp <= endTime) {
                result.add(log);
            }
        }
        return result;
    }
    
    public List<AuditLog> getAuditLogsForUser(String userId) {
        List<AuditLog> result = new ArrayList<>();
        for (AuditLog log : auditLogs) {
            if (log.userId.equals(userId)) {
                result.add(log);
            }
        }
        return result;
    }
    
    public int getActiveSessionCount() {
        cleanupExpiredSessions();
        return sessions.size();
    }
    
    private void cleanupExpiredSessions() {
        long now = System.currentTimeMillis();
        sessions.entrySet().removeIf(entry -> now - entry.getValue().createdAt > sessionTimeoutMs);
    }
    
    public Set<String> getActiveUsers() {
        cleanupExpiredSessions();
        Set<String> userIds = new HashSet<>();
        for (Session session : sessions.values()) {
            userIds.add(session.userId);
        }
        return userIds;
    }

    private record Permission(String name, String resource, String action) {}

    private record Role(String name, Set<Permission> permissions) {
        boolean hasPermission(String resource, String action) {
            for (Permission p : permissions) {
                boolean resourceMatch = p.resource.equals("*") || 
                    p.resource.equals(resource) ||
                    (p.resource.contains("*") && resource.startsWith(p.resource.replace(":*", "")));
                boolean actionMatch = p.action.equals("*") || p.action.equals(action);
                if (resourceMatch && actionMatch) {
                    return true;
                }
            }
            return false;
        }
    }

    private record User(String id, Set<Role> roles) {
        boolean hasPermission(String resource, String action) {
            for (Role role : roles) {
                if (role.hasPermission(resource, action)) {
                    return true;
                }
            }
            return false;
        }

        Set<Role> getRoles() {
            return roles;
        }
    }
    
    private record Session(String sessionId, String userId, long createdAt) {}
    
    public record AuditLog(long timestamp, String userId, String resource, String action, String detail) {
        @Override
        public String toString() {
            return String.format("AuditLog{time=%d, user=%s, resource=%s, action=%s, detail=%s}",
                timestamp, userId, resource, action, detail);
        }
    }
}