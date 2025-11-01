-- Morgan AI Assistant Database Schema
-- PostgreSQL initialization script

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- MCP Tools table
CREATE TABLE IF NOT EXISTS mcp_tools (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) UNIQUE NOT NULL,
    description TEXT NOT NULL,
    category VARCHAR(100) DEFAULT 'utility',
    endpoint_url TEXT,
    method VARCHAR(10) DEFAULT 'POST',
    auth_config JSONB,
    parameters_schema JSONB DEFAULT '{}'::jsonb,
    enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create index on name for faster lookups
CREATE INDEX IF NOT EXISTS idx_mcp_tools_name ON mcp_tools(name);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_enabled ON mcp_tools(enabled);
CREATE INDEX IF NOT EXISTS idx_mcp_tools_category ON mcp_tools(category);

-- Tool execution logs
CREATE TABLE IF NOT EXISTS tool_executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tool_id UUID REFERENCES mcp_tools(id) ON DELETE CASCADE,
    tool_name VARCHAR(255) NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    conversation_id VARCHAR(255),
    parameters JSONB,
    result JSONB,
    status VARCHAR(50) NOT NULL, -- 'success', 'error', 'timeout'
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for tool executions
CREATE INDEX IF NOT EXISTS idx_tool_executions_tool_id ON tool_executions(tool_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_user_id ON tool_executions(user_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_conversation_id ON tool_executions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_tool_executions_created_at ON tool_executions(created_at);

-- Trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_mcp_tools_updated_at BEFORE UPDATE ON mcp_tools
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

