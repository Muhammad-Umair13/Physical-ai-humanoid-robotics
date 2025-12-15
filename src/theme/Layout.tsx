import React, { type ReactNode } from 'react';
import Layout from '@theme-original/Layout';
import FrontendChatbot from '@site/src/components/FrontendChatbot';

type LayoutProps = {
  children: ReactNode;
  [key: string]: any;
};

export default function CustomLayout(props: LayoutProps): ReactNode {
  return (
    <>
      <Layout {...props}>
        {props.children}
      </Layout>
      <FrontendChatbot />
    </>
  );
}