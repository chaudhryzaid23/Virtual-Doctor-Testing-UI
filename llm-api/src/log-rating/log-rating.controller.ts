import { Response } from 'express';
import { ApiTags, ApiOperation } from '@nestjs/swagger';
import { Body, Controller, Delete, Get, Post, Res } from '@nestjs/common';

import { LogRatingService } from './log-rating.service';

@ApiTags('Log Rating')
@Controller('log-rating')
export class LogRatingController {
  constructor(private readonly logRatingService: LogRatingService) {}

  @ApiOperation({ summary: 'Append data to log file' })
  @Post('append')
  appendLog(@Body() data: any) {
    return this.logRatingService.appendLog(data);
  }

  @ApiOperation({ summary: 'Download log file' })
  @Get('download')
  downloadLog(@Res() res: Response) {
    const logPath = this.logRatingService.getLogPath();
    return res.download(logPath);
  }

  @ApiOperation({ summary: 'Delete last rating from log file' })
  @Delete('delete-last')
  deleteLastRating() {
    const result = this.logRatingService.deleteLastRating();
    return { success: result, message: 'Last rating deleted.' };
  }
}
