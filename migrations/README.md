# Database migrations

Alembic migrations are the production database schema source of truth. Set `DATABASE_URL` to a
`postgresql+psycopg://` URL and run `uv run alembic upgrade head`. Application startup never calls
`Base.metadata.create_all()`.
