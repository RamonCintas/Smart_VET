# Base clínica estruturada

Esta pasta contém dados sintéticos preparados para a fase 3 do Smart VET.

- `pacientes_clinicos.json`: 50 prontuários estruturados, identificados por `VET-001` a `VET-050`.
- `medicamentos.xlsx`: catálogo com 37 medicamentos, doses de referência, vias, contraindicações e alertas.

O backend valida os dois arquivos na inicialização, cria um índice por ID de paciente e consulta o registro atualizado a cada requisição. Em desenvolvimento, alterações nos arquivos são detectadas automaticamente.

Os PDFs fornecidos com a base não são usados em execução porque repetem o conteúdo estruturado do JSON. O benchmark de perguntas e respostas fica em `src/tests/fixtures`, separado dos dados de produção.

Apesar de sintéticos, os registros devem receber os mesmos cuidados aplicados a dados clínicos: não registrar prontuários ou prompts completos em logs e não inserir dados reais sem revisão de privacidade. O envio remoto está habilitado por padrão para esta base sintética e deve ser desabilitado com `ALLOW_REMOTE_CLINICAL_CONTEXT=false` antes do uso de dados reais.
