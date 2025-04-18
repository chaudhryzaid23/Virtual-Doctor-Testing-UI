import { Injectable, NotFoundException } from '@nestjs/common';
import * as fs from 'fs';
import * as path from 'path';
import * as csv from 'csv-parser';
import * as csvWriter from 'csv-writer';

@Injectable()
export class LogRatingService {
  private logPath = path.join(__dirname, '../../storage/llm_log.csv');

  appendLog(data: any) {
    const keys = Object.keys(data);
    const exists = fs.existsSync(this.logPath);

    const createCsvWriter = csvWriter.createObjectCsvWriter;
    const writer = createCsvWriter({
      path: this.logPath,
      header: keys.map((key) => ({ id: key, title: key })),
      append: exists,
    });

    return writer.writeRecords([data]);
  }

  getLogPath() {
    return this.logPath;
  }

  deleteLastRating(): boolean {
    if (!fs.existsSync(this.logPath)) {
      throw new NotFoundException('Log file not found.');
    }

    const data = fs.readFileSync(this.logPath, 'utf-8').split('\n');

    if (data.length <= 1) {
      // Only header or empty
      throw new NotFoundException('No ratings found to delete.');
    }

    // Remove last data row
    data.pop();

    fs.writeFileSync(this.logPath, data.join('\n'));

    return true;
  }
}
