import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
    const env = loadEnv(mode, '.', '');
    return {
      server: {
        port: 3000,
        host: '0.0.0.0',
        // 이 레포는 backend/.venv 등 파일 수가 매우 많아서
        // Vite 파일 watcher가 과부하 걸리면 "첫 화면 로딩"이 과도하게 느려질 수 있습니다.
        // 프론트와 무관한 대용량 디렉토리는 watch 대상에서 제외합니다.
        watch: {
          ignored: [
            '**/.git/**',
            '**/node_modules/**',
            '**/backend/**',
            '**/modeling/**',
            '**/bistelligence/**',
            '**/packages/**',
            '**/.venv/**',
            '**/*.pyc',
          ],
        },
      },
      // lazy import로만 등장하는 무거운 의존성은 첫 화면에서 "처음 변환"이 오래 걸릴 수 있어
      // dev 서버 시작 시 미리 prebundle 하도록 강제합니다.
      optimizeDeps: {
        include: [
          "recharts",
          "lucide-react",
        ],
      },
      plugins: [react()],
      define: {
        'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
        'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY)
      },
      resolve: {
        alias: {
          '@': path.resolve(__dirname, '.'),
        }
      }
    };
});
