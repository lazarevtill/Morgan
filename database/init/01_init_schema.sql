-- Morgan AI Assistant Database Schema
-- PostgreSQL 17 compatible

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Conversations table for persistent conversation storage
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_message_at TIMESTAMP WITH TIME ZONE,
    message_count INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT TRUE
);

CREATE INDEX IF NOT EXISTS idx_conversations_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conversations_conversation_id ON conversations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created_at ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_updated_at ON conversations(updated_at DESC);

-- Messages table for storing conversation messages
CREATE TABLE IF NOT EXISTS messages (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    sequence_number INTEGER NOT NULL,
    metadata JSONB DEFAULT '{}',
    tokens_used INTEGER,
    processing_time_ms INTEGER,
    UNIQUE(conversation_id, sequence_number)
);

CREATE INDEX IF NOT EXISTS idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX IF NOT EXISTS idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role);

-- Streaming sessions table for active streaming conversations
CREATE TABLE IF NOT EXISTS streaming_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255) NOT NULL,
    conversation_id UUID REFERENCES conversations(id) ON DELETE SET NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'paused', 'ended', 'error')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP WITH TIME ZONE,
    session_type VARCHAR(50) NOT NULL CHECK (session_type IN ('audio', 'text', 'websocket')),
    metadata JSONB DEFAULT '{}',
    audio_chunks_count INTEGER DEFAULT 0,
    total_duration_ms INTEGER DEFAULT 0
);

CREATE INDEX IF NOT EXISTS idx_streaming_sessions_session_id ON streaming_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_streaming_sessions_user_id ON streaming_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_streaming_sessions_status ON streaming_sessions(status);
CREATE INDEX IF NOT EXISTS idx_streaming_sessions_created_at ON streaming_sessions(created_at DESC);

-- Audio transcriptions table for storing STT results
CREATE TABLE IF NOT EXISTS audio_transcriptions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES streaming_sessions(id) ON DELETE CASCADE,
    message_id UUID REFERENCES messages(id) ON DELETE SET NULL,
    transcription TEXT NOT NULL,
    language VARCHAR(10),
    confidence FLOAT,
    duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    audio_format VARCHAR(50),
    sample_rate INTEGER,
    metadata JSONB DEFAULT '{}' 
);

CREATE INDEX IF NOT EXISTS idx_audio_transcriptions_session_id ON audio_transcriptions(session_id);
CREATE INDEX IF NOT EXISTS idx_audio_transcriptions_created_at ON audio_transcriptions(created_at DESC);

-- TTS generations table for caching TTS results
CREATE TABLE IF NOT EXISTS tts_generations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    message_id UUID REFERENCES messages(id) ON DELETE CASCADE,
    text_hash VARCHAR(64) NOT NULL,
    voice VARCHAR(100) NOT NULL,
    speed FLOAT DEFAULT 1.0,
    audio_data BYTEA,
    audio_format VARCHAR(50),
    sample_rate INTEGER,
    duration_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_tts_generations_text_hash ON tts_generations(text_hash);
CREATE INDEX IF NOT EXISTS idx_tts_generations_message_id ON tts_generations(message_id);
CREATE INDEX IF NOT EXISTS idx_tts_generations_created_at ON tts_generations(created_at DESC);

-- User preferences table
CREATE TABLE IF NOT EXISTS user_preferences (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) UNIQUE NOT NULL,
    preferred_language VARCHAR(10) DEFAULT 'en',
    preferred_voice VARCHAR(100) DEFAULT 'speaker_0',
    tts_speed FLOAT DEFAULT 1.0,
    llm_temperature FLOAT DEFAULT 0.7,
    llm_max_tokens INTEGER DEFAULT 2048,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_user_preferences_user_id ON user_preferences(user_id);

-- System metrics table for monitoring
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_type VARCHAR(100) NOT NULL,
    metric_value FLOAT NOT NULL,
    service_name VARCHAR(100) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_system_metrics_metric_type ON system_metrics(metric_type);
CREATE INDEX IF NOT EXISTS idx_system_metrics_service_name ON system_metrics(service_name);
CREATE INDEX IF NOT EXISTS idx_system_metrics_created_at ON system_metrics(created_at DESC);

-- Create trigger function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Add triggers for updated_at columns
CREATE TRIGGER update_conversations_updated_at
    BEFORE UPDATE ON conversations
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_streaming_sessions_updated_at
    BEFORE UPDATE ON streaming_sessions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_preferences_updated_at
    BEFORE UPDATE ON user_preferences
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create trigger to update message count in conversations
CREATE OR REPLACE FUNCTION update_conversation_message_count()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        UPDATE conversations
        SET message_count = message_count + 1,
            last_message_at = NEW.created_at
        WHERE id = NEW.conversation_id;
    ELSIF TG_OP = 'DELETE' THEN
        UPDATE conversations
        SET message_count = message_count - 1
        WHERE id = OLD.conversation_id;
    END IF;
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_conversation_message_count
    AFTER INSERT OR DELETE ON messages
    FOR EACH ROW
    EXECUTE FUNCTION update_conversation_message_count();

-- Create indexes for full-text search
CREATE INDEX idx_messages_content_trgm ON messages USING GIN (content gin_trgm_ops);
CREATE INDEX idx_conversations_title_trgm ON conversations USING GIN (title gin_trgm_ops);

-- Create views for common queries
CREATE OR REPLACE VIEW conversation_summaries AS
SELECT 
    c.id,
    c.conversation_id,
    c.user_id,
    c.title,
    c.created_at,
    c.updated_at,
    c.last_message_at,
    c.message_count,
    c.is_active,
    (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'user' ORDER BY sequence_number DESC LIMIT 1) as last_user_message,
    (SELECT content FROM messages WHERE conversation_id = c.id AND role = 'assistant' ORDER BY sequence_number DESC LIMIT 1) as last_assistant_message
FROM conversations c;

CREATE OR REPLACE VIEW active_streaming_sessions AS
SELECT 
    ss.*,
    c.conversation_id as conv_id,
    c.user_id as conv_user_id,
    (EXTRACT(EPOCH FROM (CURRENT_TIMESTAMP - ss.created_at)) * 1000)::INTEGER as session_duration_ms
FROM streaming_sessions ss
LEFT JOIN conversations c ON ss.conversation_id = c.id
WHERE ss.status = 'active';

-- Grant permissions (adjust as needed for production)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO morgan;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO morgan;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO morgan;

-- Insert default admin user preferences
INSERT INTO user_preferences (user_id, preferred_language, preferred_voice) 
VALUES ('admin', 'en', 'af_heart') 
ON CONFLICT (user_id) DO NOTHING;

-- Insert default system user preferences
INSERT INTO user_preferences (user_id, preferred_language, preferred_voice) 
VALUES ('system', 'en', 'af_heart') 
ON CONFLICT (user_id) DO NOTHING;

-- Add comments for documentation
COMMENT ON TABLE conversations IS 'Stores persistent conversation contexts';
COMMENT ON TABLE messages IS 'Stores individual messages within conversations';
COMMENT ON TABLE streaming_sessions IS 'Tracks active and historical streaming sessions';
COMMENT ON TABLE audio_transcriptions IS 'Stores STT transcription results';
COMMENT ON TABLE tts_generations IS 'Caches TTS generated audio for performance';
COMMENT ON TABLE user_preferences IS 'Stores user-specific preferences and settings';
COMMENT ON TABLE system_metrics IS 'Stores system performance and usage metrics';
