-- Morgan AI Assistant Database Schema
-- PostgreSQL 16+
-- Memory and Tools System

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";  -- For text search

-- ====================
-- MEMORIES TABLE
-- ====================
CREATE TABLE IF NOT EXISTS memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    memory_type VARCHAR(50) NOT NULL DEFAULT 'fact',  -- fact, preference, context, instruction
    category VARCHAR(100),  -- personal, work, hobby, system, etc.
    importance INTEGER DEFAULT 5 CHECK (importance BETWEEN 1 AND 10),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,  -- NULL = never expires
    access_count INTEGER DEFAULT 0,
    last_accessed_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for memories
CREATE INDEX idx_memories_user_id ON memories(user_id);
CREATE INDEX idx_memories_type ON memories(memory_type);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_importance ON memories(importance DESC);
CREATE INDEX idx_memories_created_at ON memories(created_at DESC);
CREATE INDEX idx_memories_content_trgm ON memories USING gin (content gin_trgm_ops);
CREATE INDEX idx_memories_metadata ON memories USING gin (metadata);

-- ====================
-- CONVERSATIONS TABLE
-- ====================
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    summary TEXT,
    message_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    archived_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for conversations
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX idx_conversations_archived ON conversations(archived_at) WHERE archived_at IS NOT NULL;

-- ====================
-- MESSAGES TABLE
-- ====================
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL,  -- user, assistant, system, tool
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    tokens_used INTEGER,
    processing_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for messages
CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX idx_messages_role ON messages(role);

-- ====================
-- MCP TOOLS TABLE
-- ====================
CREATE TABLE IF NOT EXISTS mcp_tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL UNIQUE,
    description TEXT,
    category VARCHAR(100),  -- utility, search, api, automation, etc.
    endpoint_url TEXT,
    method VARCHAR(10) DEFAULT 'POST',  -- GET, POST, PUT, DELETE
    auth_required BOOLEAN DEFAULT false,
    auth_config JSONB DEFAULT '{}',
    parameters_schema JSONB DEFAULT '{}',
    enabled BOOLEAN DEFAULT true,
    usage_count INTEGER DEFAULT 0,
    last_used_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for MCP tools
CREATE INDEX idx_mcp_tools_name ON mcp_tools(name);
CREATE INDEX idx_mcp_tools_category ON mcp_tools(category);
CREATE INDEX idx_mcp_tools_enabled ON mcp_tools(enabled) WHERE enabled = true;

-- ====================
-- TOOL EXECUTIONS TABLE
-- ====================
CREATE TABLE IF NOT EXISTS tool_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tool_id UUID REFERENCES mcp_tools(id) ON DELETE CASCADE,
    user_id VARCHAR(255) NOT NULL,
    conversation_id UUID REFERENCES conversations(id) ON DELETE CASCADE,
    parameters JSONB DEFAULT '{}',
    result JSONB,
    status VARCHAR(50) NOT NULL,  -- success, error, timeout, cancelled
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for tool executions
CREATE INDEX idx_tool_executions_tool_id ON tool_executions(tool_id);
CREATE INDEX idx_tool_executions_user_id ON tool_executions(user_id);
CREATE INDEX idx_tool_executions_created_at ON tool_executions(created_at DESC);
CREATE INDEX idx_tool_executions_status ON tool_executions(status);

-- ====================
-- USER PREFERENCES TABLE
-- ====================
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id VARCHAR(255) PRIMARY KEY,
    display_name VARCHAR(255),
    language VARCHAR(10) DEFAULT 'en',
    timezone VARCHAR(50) DEFAULT 'UTC',
    voice_preference VARCHAR(100),
    tts_enabled BOOLEAN DEFAULT true,
    stt_enabled BOOLEAN DEFAULT true,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- ====================
-- VECTOR REFERENCES TABLE
-- (Links PostgreSQL records to Qdrant vector IDs)
-- ====================
CREATE TABLE IF NOT EXISTS vector_references (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    qdrant_point_id UUID NOT NULL,
    qdrant_collection VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50) NOT NULL,  -- memory, message, document
    entity_id UUID NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for vector references
CREATE INDEX idx_vector_refs_point_id ON vector_references(qdrant_point_id);
CREATE INDEX idx_vector_refs_entity ON vector_references(entity_type, entity_id);
CREATE INDEX idx_vector_refs_collection ON vector_references(qdrant_collection);

-- ====================
-- ANALYTICS TABLE
-- ====================
CREATE TABLE IF NOT EXISTS analytics_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    event_data JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for analytics
CREATE INDEX idx_analytics_event_type ON analytics_events(event_type);
CREATE INDEX idx_analytics_user_id ON analytics_events(user_id);
CREATE INDEX idx_analytics_created_at ON analytics_events(created_at DESC);

-- ====================
-- TRIGGERS
-- ====================

-- Update updated_at timestamp automatically
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_memories_updated_at BEFORE UPDATE ON memories
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_conversations_updated_at BEFORE UPDATE ON conversations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_mcp_tools_updated_at BEFORE UPDATE ON mcp_tools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at BEFORE UPDATE ON user_preferences
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Update conversation message count
CREATE OR REPLACE FUNCTION update_conversation_message_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE conversations
        SET message_count = message_count + 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = NEW.conversation_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE conversations
        SET message_count = message_count - 1,
            updated_at = CURRENT_TIMESTAMP
        WHERE id = OLD.conversation_id;
    END IF;
    RETURN NULL;
END;
$$ language 'plpgsql';

CREATE TRIGGER trigger_conversation_message_count
    AFTER INSERT OR DELETE ON messages
    FOR EACH ROW EXECUTE FUNCTION update_conversation_message_count();

-- ====================
-- VIEWS
-- ====================

-- Recent memories view
CREATE OR REPLACE VIEW recent_memories AS
SELECT
    m.*,
    v.qdrant_point_id,
    v.qdrant_collection
FROM memories m
LEFT JOIN vector_references v ON v.entity_id = m.id AND v.entity_type = 'memory'
WHERE m.expires_at IS NULL OR m.expires_at > CURRENT_TIMESTAMP
ORDER BY m.created_at DESC;

-- Tool usage statistics
CREATE OR REPLACE VIEW tool_usage_stats AS
SELECT
    t.id,
    t.name,
    t.category,
    t.usage_count AS total_uses,
    COUNT(te.id) AS recent_executions,
    AVG(te.execution_time_ms) AS avg_execution_time,
    SUM(CASE WHEN te.status = 'success' THEN 1 ELSE 0 END) AS success_count,
    SUM(CASE WHEN te.status = 'error' THEN 1 ELSE 0 END) AS error_count
FROM mcp_tools t
LEFT JOIN tool_executions te ON te.tool_id = t.id
    AND te.created_at > CURRENT_TIMESTAMP - INTERVAL '30 days'
GROUP BY t.id, t.name, t.category, t.usage_count;

-- User activity summary
CREATE OR REPLACE VIEW user_activity_summary AS
SELECT
    user_id,
    COUNT(DISTINCT c.id) AS total_conversations,
    COUNT(m.id) AS total_messages,
    COUNT(DISTINCT mem.id) AS total_memories,
    MAX(c.updated_at) AS last_conversation_at,
    MAX(mem.created_at) AS last_memory_at
FROM conversations c
LEFT JOIN messages m ON m.conversation_id = c.id
LEFT JOIN memories mem ON mem.user_id = c.user_id
GROUP BY user_id;

-- ====================
-- INITIAL DATA
-- ====================

-- Insert default system tools
INSERT INTO mcp_tools (name, description, category, enabled) VALUES
    ('web_search', 'Search the web for information', 'search', true),
    ('calculator', 'Perform mathematical calculations', 'utility', true),
    ('datetime', 'Get current date and time information', 'utility', true),
    ('weather', 'Get weather information', 'api', false),
    ('reminder', 'Create reminders', 'automation', false)
ON CONFLICT (name) DO NOTHING;

-- Grant permissions (if using specific users)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO morgan;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO morgan;

-- Success message
DO $$
BEGIN
    RAISE NOTICE 'Morgan AI Assistant database schema initialized successfully!';
END $$;
