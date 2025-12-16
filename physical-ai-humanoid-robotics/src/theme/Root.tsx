import React, { type ReactNode } from 'react';
import FrontendChatbot from '@site/src/components/FrontendChatbot';

type RootProps = {
  children: ReactNode;
};

export default function Root({ children }: RootProps): ReactNode {
  return (
    <>
      {children}
      <FrontendChatbot />
    </>
  );
}