import psycopg2

class Store:
    def __init__(self, host: str = "localhost", 
                 port: int = 5432,
                 user: str = "postgres",
                 password: str = "postgres",
                 database: str = "postgres"):
        self.connection_params = {
            "host": host,
            "port": port,
            "user": user,
            "password": password,
            "database": database
        }
        self.conn = None

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
                               random() * power(1.1, j.reaction_count) as weight
                        FROM jokes j
                        JOIN messages m1 ON j.source_message_id = m1.message_id
                        JOIN messages m2 ON j.joke_message_id = m2.message_id
                    )
                    SELECT source_content, joke_content 
                    FROM weighted_jokes
                    ORDER BY weight DESC
                    LIMIT %s
                    """,
                    (n,)
                )
                return cur.fetchall()
        except Exception as e:
            print(f"Error fetching random jokes: {e}")
            return []

    def __del__(self):
        if hasattr(self, 'conn'):
            self.conn.close()
