import psycopg2
from dataclasses import dataclass

@dataclass
class GuildConfig:
    guild_id: int
    archive_channel_id: int = 0
    delete_jokes_after_minutes: int = 0
    downvote_reaction_threshold: int = 0
    enable_country_jokes: bool = True

class Store:
    def __init__(self, host: str = "localhost", 
                 port: int = 5432,
                 user: str = "postgres",
                 password: str = "postgres",
                 database: str = "postgres",
                 weight_coef: float = 1.1):
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }
        self.conn = None
        self.weight_coef = weight_coef
        self._guild_configs = {}  # Add cache dictionary

    def _connect(self):
        return psycopg2.connect(**self.connection_params)

    def _ensure_connection(self):
        try:
            if self.conn is None:
                self.conn = self._connect()
                return
                
            # Test if connection is still alive
            with self.conn.cursor() as cur:
                cur.execute("SELECT 1")
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            try:
                # Close the dead connection if it exists
                if self.conn is not None:
                    self.conn.close()
            except:
                pass
            # Create new connection
            self.conn = self._connect()
    
    def save(self, source_message_id: int, 
            joke_message_id: int,
            source_message_content: str,
            joke_message_content: str,
            reaction_count: int = 0) -> None:
        
        # Print joke data
        joke_data = {
            "source_message_id": source_message_id,
            "joke_message_id": joke_message_id,
            "source_message_content": source_message_content,
            "joke_message_content": joke_message_content,
            "reaction_count": reaction_count
        }
        print(f"Store saving joke: {joke_data}")
        
        try:
            self._ensure_connection()
            with self.conn.cursor() as cur:
                # Insert both messages with content update on conflict
                cur.execute(
                    """
                    INSERT INTO messages (message_id, content) 
                    VALUES (%s, %s)
                    ON CONFLICT (message_id) 
                    DO UPDATE SET content = EXCLUDED.content
                    """,
                    (source_message_id, source_message_content)
                )
                cur.execute(
                    """
                    INSERT INTO messages (message_id, content) 
                    VALUES (%s, %s)
                    ON CONFLICT (message_id) 
                    DO UPDATE SET content = EXCLUDED.content
                    """,
                    (joke_message_id, joke_message_content)
                )
                
                # Create or update relationship in jokes table
                cur.execute(
                    """
                    INSERT INTO jokes (source_message_id, joke_message_id, reaction_count) 
                    VALUES (%s, %s, %s)
                    ON CONFLICT (source_message_id, joke_message_id) 
                    DO UPDATE SET reaction_count = EXCLUDED.reaction_count
                    """,
                    (source_message_id, joke_message_id, reaction_count)
                )
                
                self.conn.commit()
        except Exception as e:
            self.conn.rollback()
            raise e

    def get_random_jokes(self, n: int) -> list[tuple[str, str]]:
        try:
            self._ensure_connection()
            with self.conn.cursor() as cur:
                cur.execute(
                    """
                    WITH weighted_jokes AS (
                        SELECT m1.content as source_content,
                               m2.content as joke_content,
                               random() * power(%s, j.reaction_count) as weight
                        FROM jokes j
                        JOIN messages m1 ON j.source_message_id = m1.message_id
                        JOIN messages m2 ON j.joke_message_id = m2.message_id
                    )
                    SELECT source_content, joke_content 
                    FROM weighted_jokes
                    ORDER BY weight DESC
                    LIMIT %s
                    """,
                    (self.weight_coef, n)
                )
                return cur.fetchall()
        except Exception as e:
            print(f"Error fetching random jokes: {e}")
            return []

    def get_guild_config(self, guild_id: int) -> GuildConfig:
        if guild_id not in self._guild_configs:
            self._ensure_connection()
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO guild_configs (guild_id) VALUES (%s) ON CONFLICT DO NOTHING",
                    (guild_id,)
                )
                
                cur.execute(
                    """
                    SELECT archive_channel_id, delete_jokes_after_minutes, 
                           downvote_reaction_threshold, enable_country_jokes 
                    FROM guild_configs 
                    WHERE guild_id = %s
                    """,
                    (guild_id,)
                )
                result = cur.fetchone()
                self.conn.commit()
                
                self._guild_configs[guild_id] = GuildConfig(
                    guild_id=guild_id,
                    archive_channel_id=result[0],
                    delete_jokes_after_minutes=result[1],
                    downvote_reaction_threshold=result[2],
                    enable_country_jokes=result[3]
                )
        
        return self._guild_configs[guild_id]

    def save_guild_config(self, config: GuildConfig) -> None:
        self._ensure_connection()
        with self.conn.cursor() as cur:
            cur.execute(
                """
                UPDATE guild_configs 
                SET archive_channel_id = %s,
                    delete_jokes_after_minutes = %s,
                    downvote_reaction_threshold = %s,
                    enable_country_jokes = %s
                WHERE guild_id = %s
                """,
                (config.archive_channel_id, config.delete_jokes_after_minutes,
                 config.downvote_reaction_threshold, config.enable_country_jokes,
                 config.guild_id)
            )
            self.conn.commit()
            # Update cache
            self._guild_configs[config.guild_id] = config

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
