# miniRAG

Тривиальный RAG-пайплайн, отвечающий на вопросы по документации библиотеки numpy.

pdf_processor.py - парсит pdf-документацию в txt;

answer_questions.py - архитектура RAG-пайплайна;

evaluate.py - бенчмарк архитектуры на questions.json.

------------------------------------------

base_config.json - конфиг RAG-пайплайна, сериализуется в pydantic-структуру.

questions.json - набор вопросов для бенчмарка
