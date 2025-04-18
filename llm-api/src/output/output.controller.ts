import {
  Controller,
  Post,
  UploadedFile,
  UseInterceptors,
  Get,
  Param,
  Res,
} from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { OutputService } from './output.service';
import { Response } from 'express';
import * as path from 'path';

@Controller('output-files')
export class OutputController {
  constructor(private readonly outputService: OutputService) {}

  @Post('upload')
  @UseInterceptors(
    FileInterceptor('file', {
      storage: diskStorage({
        destination: './storage/outputs',
        filename: (req, file, cb) => {
          cb(null, `${Date.now()}_${file.originalname}`);
        },
      }),
    }),
  )
  uploadFile(@UploadedFile() file: Express.Multer.File) {
    return { message: 'File uploaded successfully', filename: file.filename };
  }

  @Get('download/:filename')
  downloadOutput(@Param('filename') filename: string, @Res() res: Response) {
    const filePath = path.join(__dirname, '../../storage/outputs', filename);
    return res.download(filePath);
  }
}
